import logging
import socket
from math import atan2, sqrt

import numpy as np
from pylivelinkface import FaceBlendShape, PyLiveLinkFace

from src.types import FrameResult

logger = logging.getLogger(__name__)

# Map our camelCase blendshape names to PyLiveLinkFace PascalCase enum members.
# Built dynamically: "eyeBlinkLeft" -> FaceBlendShape.EyeBlinkLeft
_BLENDSHAPE_MAP: dict[str, FaceBlendShape] = {}
for _member in FaceBlendShape:
    # Convert PascalCase enum name to camelCase for lookup
    _camel = _member.name[0].lower() + _member.name[1:]
    _BLENDSHAPE_MAP[_camel] = _member


def _rotation_matrix_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw, pitch, roll (radians) from a 4x4 transformation matrix."""
    R = matrix[:3, :3]
    sy = sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = atan2(-R[2, 0], sy)
        yaw = atan2(R[1, 0], R[0, 0])
        roll = atan2(R[2, 1], R[2, 2])
    else:
        pitch = atan2(-R[2, 0], sy)
        yaw = atan2(-R[1, 2], R[1, 1])
        roll = 0.0
    return yaw, pitch, roll


class LiveLinkSender:
    """Sends facial blendshape data to UE5 via Live Link Face UDP protocol."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        subject_name: str = "PythonFace",
        fps: int = 60,
    ):
        self._host = host
        self._port = port
        self._face = PyLiveLinkFace(name=subject_name, fps=fps, filter_size=1)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"Live Link sender targeting {host}:{port} (subject: {subject_name})")

    def send_frame(self, result: FrameResult) -> None:
        """Send one frame of blendshape data to UE5."""
        if not result.blendshapes:
            return

        bs = result.blendshapes[0]  # First face

        # Set 52 ARKit blendshapes
        for name, value in bs.items():
            if name == "_neutral":
                continue
            enum_member = _BLENDSHAPE_MAP.get(name)
            if enum_member is not None:
                self._face.set_blendshape(enum_member, value, no_filter=True)

        # Set head rotation from transformation matrix
        if result.transformation_matrix and result.transformation_matrix[0] is not None:
            yaw, pitch, roll = _rotation_matrix_to_euler(result.transformation_matrix[0])
            self._face.set_blendshape(FaceBlendShape.HeadYaw, yaw, no_filter=True)
            self._face.set_blendshape(FaceBlendShape.HeadPitch, pitch, no_filter=True)
            self._face.set_blendshape(FaceBlendShape.HeadRoll, roll, no_filter=True)

        self._sock.sendto(self._face.encode(), (self._host, self._port))

    def close(self) -> None:
        self._sock.close()
        logger.info("Live Link sender closed")
