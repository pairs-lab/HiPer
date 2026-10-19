"""
hiper.segmentation.seg_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Segmentation backend that wraps MobileSAM + CLIP/DINOv2.
Supports two modes:
  - Direct mode : runs inference in-process (requires Python 3.8
                  compatible model versions)
  - Socket mode  : forwards the image to a Python 3.12 inference
                   server over TCP (for Jetson + RTX 5060 Blackwell,
                   where sm_120 CUDA is incompatible with ROS Python).
                   This is the same socket-bridge architecture used in
                   your existing SLIDE_SLAM setup.
"""

import json
import socket
import struct
import numpy as np
import rospy

from sensor_msgs.msg import Image as ROSImage

try:
    from cv_bridge import CvBridge
    import cv2
    _CV_OK = True
except ImportError:
    _CV_OK = False


class SegmentationBackend:
    """
    Unified interface for segmentation regardless of execution mode.
    """

    def __init__(
        self,
        model_name: str = "mobile_sam",
        clip_model: str = "ViT-B/32",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        max_instances: int = 32,
        use_socket: bool = False,
        socket_host: str = "localhost",
        socket_port: int = 9876,
    ):
        self._model_name  = model_name
        self._device      = device
        self._conf_thresh = confidence_threshold
        self._max_inst    = max_instances
        self._use_socket  = use_socket
        self._bridge      = CvBridge() if _CV_OK else None

        if use_socket:
            self._sock_host = socket_host
            self._sock_port = socket_port
            rospy.loginfo(
                f"[seg_backend] Socket mode → {socket_host}:{socket_port}")
        else:
            self._model = self._load_model(model_name, clip_model, device)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, img_msg: ROSImage) -> dict:
        """
        Run segmentation on a ROS Image message.

        Returns a dict:
          "mask"             : np.ndarray (H, W) uint16 — instance ids
          "embeddings"       : np.ndarray (N, embed_dim) float32
          "instance_classes" : dict {str(inst_id): class_label}
          "instance_scores"  : dict {str(inst_id): confidence}
          "num_instances"    : int
        """
        if self._use_socket:
            return self._process_socket(img_msg)
        else:
            return self._process_direct(img_msg)

    def render_overlay(self, img_msg: ROSImage, result: dict) -> ROSImage:
        """Draw coloured instance masks on the RGB image for RViz."""
        if not _CV_OK or self._bridge is None:
            return img_msg

        cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8").copy()
        mask   = result["mask"]
        H, W   = mask.shape

        np.random.seed(42)
        colours = {
            i: tuple(int(c) for c in np.random.randint(0, 255, 3))
            for i in np.unique(mask) if i > 0
        }
        overlay = cv_img.copy()
        for inst_id, colour in colours.items():
            overlay[mask == inst_id] = colour

        blended = cv2.addWeighted(cv_img, 0.5, overlay, 0.5, 0)

        out_msg = self._bridge.cv2_to_imgmsg(blended, "bgr8")
        out_msg.header = img_msg.header
        return out_msg

    # ── Direct mode ───────────────────────────────────────────────────────────

    def _process_direct(self, img_msg: ROSImage) -> dict:
        """Run inference in-process."""
        if not _CV_OK or self._bridge is None:
            return self._empty_result(img_msg)

        cv_img = self._bridge.imgmsg_to_cv2(img_msg, "rgb8")
        H, W   = cv_img.shape[:2]

        if self._model is None:
            return self._empty_result(img_msg)

        try:
            masks_list, scores, class_labels, embeddings = \
                self._model.segment(cv_img)
        except Exception as e:
            rospy.logwarn_throttle(5.0,
                f"[seg_backend] Inference error: {e}")
            return self._empty_result(img_msg)

        return self._pack_result(H, W, masks_list, scores,
                                 class_labels, embeddings)

    # ── Socket mode ───────────────────────────────────────────────────────────

    def _process_socket(self, img_msg: ROSImage) -> dict:
        """
        Forward image bytes to the Python 3.12 inference server.
        Protocol (same as SLIDE_SLAM socket bridge):
          → 4 bytes: payload length (big-endian uint32)
          → N bytes: JSON {"h":H, "w":W, "enc":encoding,
                           "data": base64-encoded image bytes}
          ← 4 bytes: response length
          ← M bytes: JSON result
        """
        if not _CV_OK:
            return self._empty_result(img_msg)

        import base64

        H = img_msg.height
        W = img_msg.width
        payload = json.dumps({
            "h":    H,
            "w":    W,
            "enc":  img_msg.encoding,
            "data": base64.b64encode(bytes(img_msg.data)).decode("ascii"),
            "conf": self._conf_thresh,
            "max_inst": self._max_inst,
        }).encode("utf-8")

        try:
            with socket.create_connection(
                    (self._sock_host, self._sock_port), timeout=2.0) as sock:
                # Send
                sock.sendall(struct.pack(">I", len(payload)) + payload)
                # Receive
                raw_len = self._recv_exactly(sock, 4)
                resp_len = struct.unpack(">I", raw_len)[0]
                resp_bytes = self._recv_exactly(sock, resp_len)
                resp = json.loads(resp_bytes.decode("utf-8"))
        except Exception as e:
            rospy.logwarn_throttle(
                5.0, f"[seg_backend] Socket error: {e}")
            return self._empty_result(img_msg)

        # Decode response
        mask_flat   = np.array(resp["mask"], dtype=np.uint16)
        mask        = mask_flat.reshape(H, W)
        embeddings  = np.array(resp["embeddings"],
                               dtype=np.float32).reshape(
                                   resp["num_instances"], -1) \
                      if resp["num_instances"] > 0 \
                      else np.zeros((0, 512), dtype=np.float32)

        return {
            "mask":             mask,
            "embeddings":       embeddings,
            "instance_classes": resp["instance_classes"],
            "instance_scores":  resp["instance_scores"],
            "num_instances":    resp["num_instances"],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _recv_exactly(sock: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed unexpectedly.")
            buf += chunk
        return buf

    @staticmethod
    def _pack_result(H, W, masks_list, scores,
                     class_labels, embeddings) -> dict:
        mask = np.zeros((H, W), dtype=np.uint16)
        for i, m in enumerate(masks_list, start=1):
            mask[m > 0] = i

        inst_classes = {str(i+1): lbl
                        for i, lbl in enumerate(class_labels)}
        inst_scores  = {str(i+1): float(s)
                        for i, s in enumerate(scores)}
        return {
            "mask":             mask,
            "embeddings":       np.array(embeddings, dtype=np.float32)
                                if len(embeddings) > 0
                                else np.zeros((0, 512), dtype=np.float32),
            "instance_classes": inst_classes,
            "instance_scores":  inst_scores,
            "num_instances":    len(masks_list),
        }

    @staticmethod
    def _empty_result(img_msg: ROSImage) -> dict:
        H = img_msg.height
        W = img_msg.width
        return {
            "mask":             np.zeros((H, W), dtype=np.uint16),
            "embeddings":       np.zeros((0, 512), dtype=np.float32),
            "instance_classes": {},
            "instance_scores":  {},
            "num_instances":    0,
        }

    @staticmethod
    def _load_model(model_name, clip_model, device):
        """
        Attempt to load the segmentation model.
        Returns None and logs a warning if unavailable.
        """
        try:
            if model_name == "mobile_sam":
                from hiper.segmentation.mobile_sam_wrapper \
                    import MobileSAMWithCLIP
                return MobileSAMWithCLIP(clip_model=clip_model, device=device)
        except ImportError as e:
            rospy.logwarn(
                f"[seg_backend] Could not load {model_name}: {e}. "
                "Segmentation node will publish empty masks.")
        return None
