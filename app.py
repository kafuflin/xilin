import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import torch  # 新增PyTorch导入

# 设置页面布局
st.set_page_config(
    page_title="YOLOv8 实时检测",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# 标题
st.title("YOLOv8 中药目标检测识别")

# 侧边栏设置
with st.sidebar:
    st.header("配置")

    # 上传自定义模型
    model_file = st.file_uploader("上传YOLOv8模型 (.pt)", type=["pt"])

    # 选择输入源
    input_type = st.radio("选择输入类型", ["图片", "视频", "摄像头"])

    # 置信度阈值
    conf = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)

    # NMS IoU阈值
    iou = st.slider("NMS IoU阈值", 0.0, 1.0, 0.45, 0.01)  # 新增参数

    # 保存结果开关
    save_output = st.checkbox("保存处理结果")


# 加载模型
@st.cache_resource
def load_model(model_path):
    try:
        # 检查GPU可用性
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.success(f"使用设备: {device.upper()}")

        # 加载模型并指定设备
        model = YOLO(model_path, task='detect').to(device)
        return model
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None


# 改进后的图片处理函数
def process_image(model, img, conf, iou):
    # 转换颜色空间 BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 执行推理（添加缺失参数）
    results = model.predict(
        img_rgb,
        imgsz=640,
        conf=conf,
        iou=iou,
        device=model.device  # 使用模型所在设备
    )

    # 生成标注图像（BGR格式）
    annotated_img = results[0].plot()
    return annotated_img[..., ::-1]  # RGB转BGR


# 改进后的视频帧处理函数
def process_frame(model, frame, conf, iou):
    # 转换颜色空间 BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 执行推理
    results = model.predict(
        frame_rgb,
        imgsz=640,
        conf=conf,
        iou=iou,
        device=model.device,
        verbose=False  # 关闭冗余输出
    )

    # 生成标注图像
    return results[0].plot()[..., ::-1]


# 主界面
col1, col2 = st.columns(2)

if model_file is not None:
    # 保存上传的模型到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_file.write(model_file.getvalue())
        model_path = tmp_file.name

    # 加载模型
    model = load_model(model_path)

    if input_type == "图片":
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # 读取并缓存图片
            image = Image.open(uploaded_file)
            img = np.array(image.convert('RGB'))

            # 显示原始图片（不受按钮控制）
            with col1:
                st.image(image, caption="原始图片", use_column_width=True)

            # 添加检测按钮
            if st.button("开始检测", key="detect_image"):
                # 处理图片（传入iou参数）
                processed_img = process_image(model, img, conf, iou)

                with col2:
                    # 显示检测信息
                    results = model(img)
                    detection_info = ""
                    for box in results[0].boxes:
                        cls_name = model.names[int(box.cls)]
                        conf_val = box.conf.item()
                        detection_info += f"**{cls_name}**: {conf_val:.2f}\n"

                    # 分栏显示结果
                    col2a, col2b = st.columns([3, 1])
                    with col2a:
                        st.image(processed_img, caption="检测结果", use_column_width=True)
                    with col2b:
                        st.markdown("**检测目标置信度**")
                        st.markdown(detection_info)

                    if save_output:
                        output = Image.fromarray(processed_img[..., ::-1])
                        st.download_button(
                            label="下载结果",
                            data=output,
                            file_name="processed_image.png",
                            mime="image/png"
                        )

    # 视频和摄像头部分保持不变...

    elif input_type == "视频":
        uploaded_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # 保存视频到临时文件
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)

            # 获取视频信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 设置视频输出
            if save_output:
                output_path = "processed_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 视频显示
            video_placeholder = st.empty()
            stop_button = st.button("停止处理")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧（传入iou参数）
                processed_frame = process_frame(model, frame, conf, iou)

                # 显示处理后的帧
                video_placeholder.image(processed_frame, channels="BGR")

                # 保存结果
                if save_output:
                    out.write(processed_frame)

            cap.release()
            if save_output:
                out.release()
                st.success(f"处理后的视频已保存到 {output_path}")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="下载处理后的视频",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

    elif input_type == "摄像头":
        cam_placeholder = st.empty()
        stop_button = st.button("停止摄像头")

        if st.button("开启摄像头"):
            st.session_state.camera_on = True

        if st.session_state.camera_on:
            cap = cv2.VideoCapture(0)

            while cap.isOpened() and st.session_state.camera_on and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("无法读取摄像头")
                    break

                # 处理帧（传入iou参数）
                processed_frame = process_frame(model, frame, conf, iou)

                # 显示处理后的帧
                cam_placeholder.image(processed_frame, channels="BGR")

            cap.release()
            st.session_state.camera_on = False

# 清理临时文件
if model_file is not None:
    os.unlink(model_path)
