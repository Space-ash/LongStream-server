import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 自动下载并加载最轻量的 YOLO11 实例分割模型
    print("正在加载模型...")
    model = YOLO('./model/yolo11n-seg.pt')
    
    # 打开本地默认摄像头 (0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("开始实时处理，按 'q' 键退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 运行推理，我们暂时只识别“人 (class 0)”进行测试
        results = model(frame, classes=[0], verbose=False)
        
        output_frame = frame.copy()
        
        if results[0].masks is not None:
            # 提取掩码并合并
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]))
            
            # 将人的区域涂黑
            output_frame[combined_mask > 0] = [0, 0, 0]
            
            # 为了方便观察，我们把掩码区域用红色半透明高亮出来 (可选)
            # output_frame[combined_mask > 0] = output_frame[combined_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
            
        # 并排显示：左边原图，右边涂黑后的图
        display_img = np.hstack((frame, output_frame))
        cv2.imshow("YOLO11 Dynamic Masking Test", display_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()