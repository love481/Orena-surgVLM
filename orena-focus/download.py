from focus import FocusConfig, set_config, download
from focus.preprocessing import (
    VideoTimestampOverlayPreprocessor,
    FrameExtractorPreprocessor,
)

set_config(FocusConfig(root_dir="/iopsstor/scratch/cscs/lpanta32/focus"))
# download("lapchole")
download("heico")
# FrameExtractorPreprocessor(stride=1).process(dataset="heico")

VideoTimestampOverlayPreprocessor().process(dataset="heico")
# VideoTimestampOverlayPreprocessor().process(dataset="lapchole")