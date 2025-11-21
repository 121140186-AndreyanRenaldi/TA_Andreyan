from dataclasses import dataclass
import numpy as np

@dataclass
class SchemeConfig:
    name: str
    scheme_type: str           
    video_path: str
    roi1: np.ndarray
    roi2: np.ndarray
    scale: float                
    use_union_mask: bool = True
    height: int = 10              
    width: int = 12               
    min_contour_area: int = 2000
    max_lost: int = 15
    match_distance: float = 100.0

SCHEME_1 = SchemeConfig(
    name="tengah_7",
    scheme_type="1d",
    video_path="../assets/video/TENGAH-7.MOV",
    roi1=np.array(
        [[(426, 849), (638, 576), (1190, 584), (1510, 856)]], np.int32
    ),
    roi2=np.array(
        [[(638, 576), (692, 507), (1105, 515), (1190, 584)]], np.int32
    ),
    scale=0.85,
)

SCHEME_2 = SchemeConfig(
    name="tengah_9",
    scheme_type="1d",
    video_path="../assets/video/TENGAH-9.MOV",
    roi1=np.array(
        [[(498, 982), (682, 646), (1222, 655), (1525, 993)]], np.int32
    ),
    roi2=np.array(
        [[(682, 646), (734, 557), (1136, 565), (1222, 655)]], np.int32
    ),
    scale=0.85,
)

SCHEME_3 = SchemeConfig(
    name="kiri_7",
    scheme_type="2d",
    video_path="../assets/video/KIRI-7.MOV",
    roi1 = np.array(
        [[(147, 997), (93, 644), (720, 587), (1180, 786)]], np.int32
    ),
    roi2=np.array(
        [[(93, 644), (82, 555), (580,527), (720, 587)]], np.int32
    ),
    scale=0.1,
)

SCHEME_4 = SchemeConfig(
    name="kiri_9",
    scheme_type="2d",
    video_path="../assets/video/KIRI-9.MOV",
    roi1=np.array(
        [[(129, 1039), (79, 678), (675, 613), (1067, 827)]], np.int32
    ),
    roi2=np.array(
        [[(79, 678), (0,0), (0,0), (675, 613)]], np.int32
    ),
    scale=1.0,
)

SCHEMES = {
    "tengah_7": SCHEME_1,
    "tengah_9": SCHEME_2,
    "kiri_7": SCHEME_3,
    "kiri_9": SCHEME_4,
}

SCHEME_ORDER = {
    1: "tengah_7",
    2: "tengah_9",
    3: "kiri_7",
    4: "kiri_9",
}
