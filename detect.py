import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import datetime
import time

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from threading import Thread
import subprocess as sp
class YOUTUBE_RTMP:
    '''edit sjs, added YOUTUBE_RTMP class'''
    def __init__(self,YOUTUBE_STREAM_KEY):
        self.YOUTUBE_STREAM_KEY=YOUTUBE_STREAM_KEY
        self.initiate=True
    def Preprocess(self,HEIGHT_i,WIDTH_i,VBR_i):
        self.HEIGHT_i=HEIGHT_i
        self.WIDTH_i=WIDTH_i
        self.VBR_i=VBR_i     
        self.initiate=False
        self.startFFmpeg_Process()
    def startFFmpeg_Process(self):
        self.cmd=["ffmpeg","-y","-f","lavfi","-i","anullsrc","-f","rawvideo","-vcodec","rawvideo", "-s","{}x{}".format(self.HEIGHT_i,self.WIDTH_i),
        "-pix_fmt","bgr24","-i","-","-acodec","aac","-ar","44100","-b:a","712000","-vcodec","libx264","-preset","medium","-b:v","{}".format(self.VBR_i),"-bufsize","0","-pix_fmt",
        "yuv420p","-f","flv","-crf","18","rtmp://a.rtmp.youtube.com/live2/{}".format(self.YOUTUBE_STREAM_KEY)]
        self.process = sp.Popen(self.cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    def write(self,frame,VBR_i):
        # initiate FFmpeg process on first run
        if self.initiate:
            # start pre-processing and initiate process
            self.Preprocess(frame.shape[1],frame.shape[0],VBR_i)
            # Check status of the process
            assert self.process is not None
        # write the frame
        try:
            self.process.stdin.write(frame.tostring())
        except (OSError, IOError):
            # log something is wrong!
            print(
                "BrokenPipeError caught"
            )
            raise ValueError  # for testing purpose only
    def close(self):
        if self.process.stdin:
            self.process.stdin.close()  # close `stdin` output
        self.process.wait()  # wait if still process is still processing some information
        self.process = None
  

def YOUTUBE_STREAM_RESOLUTION(res='720p'):
    '''edit sjs, added YOUTUBE_STREAM_RESOLUTION function'''
    #returns the height,width, and video bit rate
    if res=='720p':
        return 720,1280,'4000k'
    elif res=='1080p':
        return 1080,1920,'6000k'
    elif res=='480p':
        return 480,854,'2000k'
    elif res=='360p':
        return 640,360,'1000k'
    else:
        print('DID NOT RECOGNIZE res=={}\n so using res==720p'.format(res))
        return 720,1280,'4000k'
def send_imgs(sender,im0):
    try:
        sender.send_image("YOLO OUPUT", im0)
    except:
        pass
def run_cmd(cmd_i):
    os.system(cmd_i)
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    #edit sjs
    YOUTUBE_STREAM_KEY = opt.YOUTUBE_RTMP
    YOUTUBE_STREAM_RES = opt.YOUTUBE_STREAM_RES
    if YOUTUBE_STREAM_KEY!='xxxx-xxxx-xxxx-xxxx-xxxx':
        RTMP=True
        writer=YOUTUBE_RTMP(YOUTUBE_STREAM_KEY)
    else:
        RTMP=False

    #Sending images to cell phone via text/email
    send_image_to_cell=opt.send_image_to_cell
    send_image_to_cell_path=opt.send_image_to_cell_path
    destinations=opt.destinations
    basepath_chips=opt.basepath_chips
    send_allowed=True
    sleep_time_chips=opt.sleep_time_chips
    start_time=time.time()

    date_i=str(datetime.datetime.now()).replace(' ','_').replace('.','p').replace(':',"c").replace('-','_')
    if os.path.exists(basepath_chips)==False and send_image_to_cell:
        if os.path.exists(os.path.dirname(basepath_chips)):
            os.makedirs(basepath_chips)
            basepath_chips=os.path.join(basepath_chips,date_i)
            if os.path.exists(basepath_chips)==False:
                os.makedirs(basepath_chips)
        else:
            send_image_to_cell=False
            print('You have a bad path to save chips.  Not sending images to cell')
    elif send_image_to_cell:
        basepath_chips=os.path.join(basepath_chips,date_i)
        if os.path.exists(basepath_chips)==False:
            os.makedirs(basepath_chips)


    RTSP_PATH = opt.RTSP_PATH
    RTSP_SERVER_PATH=opt.RTSP_SERVER_PATH
    RTSP=False
    running=False
    if RTSP_PATH != 'xxxx-xxxx-xxxx-xxxx-xxxx' and os.path.exists(RTSP_SERVER_PATH):
        print(RTSP_SERVER_PATH)
        if opt.width:
            WIDTH=opt.width
        else:
            WIDTH=imgsz
        if opt.height:
            HEIGHT=opt.height
        else:
            HEIGHT=imgsz
        import sys
        #from threading import Thread
        sys.path.append(os.path.dirname(RTSP_SERVER_PATH))
        #import rtsp_server as rs
        import imagezmq
        sender = imagezmq.ImageSender()
        RTSP=True
        
        #cmd_i='python3 {} --fps=30 --width={} --height={} --port={} --stream_key={}'.format(RTSP_SERVER_PATH, WIDTH,HEIGHT,opt.port,opt.stream_key)
        #RunMe=Thread(target=run_cmd,args=(cmd_i,)).start()
        #RunMe=Thread(target=rs.RunMe,args=(30,WIDTH,HEIGHT,opt.port,opt.stream_key,)).start()
        
    
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #Check send chips
                time_now=time.time()
                if time_now-start_time>sleep_time_chips:
                    send_allowed=True
                    start_time=time_now
                else:
                    send_allowed=False

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    
                    if send_image_to_cell and os.path.exists(send_image_to_cell_path) and send_allowed:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        label=str(datetime.datetime.now())+"_"+label
                        label=label.replace(' ','_').replace('.','p').replace(':',"c").replace('-','_')
                        chip_i=label+".jpg"
                        chip_i=os.path.join(basepath_chips,chip_i)
                        print('im0.shape',im0.shape)
                        print('chip_i',chip_i)
                        boxes=xyxy
                        print('boxes',boxes)
                        #print(int(boxes[0].cpu().detach().numpy()),int(boxes[2].cpu().detach().numpy()),int(boxes[1].cpu().detach().numpy()),int(boxes[3].cpu().detach().numpy()))
                        cv2.imwrite(chip_i,im0[int(boxes[1].cpu().detach().numpy()):int(boxes[3].cpu().detach().numpy()),int(boxes[0].cpu().detach().numpy()):int(boxes[2].cpu().detach().numpy()),:])
                        main_message="Detected {}".format(label)
                        cmd_i='python3 {} --destinations={} --main_message="{}"  --img_path="{}" '.format(send_image_to_cell_path,destinations,main_message,chip_i)
                        print(cmd_i)
                        Thread(target=run_cmd,args=(cmd_i,)).start()

            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w, h = im0.shape[1], im0.shape[0] #edit sjs
                            
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            # Send to YOUTUBE (image with detections)
            if RTMP:
                YH_i,YW_i,VBR_i=YOUTUBE_STREAM_RESOLUTION(res=YOUTUBE_STREAM_RES)
                image=cv2.resize(im0,(YW_i,YH_i))
                writer.write(image,VBR_i)
            if running and RTSP:
                running=True
                pass
            elif RTSP:
                #input("RUNNING")
                cmd_i='python3 {} --fps=30 --width={} --height={} --port={} --stream_key={}'.format(RTSP_SERVER_PATH, im0.shape[1],im0.shape[0],opt.port,opt.stream_key)
                RunMe=Thread(target=run_cmd,args=(cmd_i,)).start()
                running=True
            if RTSP:
                Thread(target=send_imgs,args=(sender,im0,)).start()
                

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument("--YOUTUBE_RTMP",type=str,default="xxxx-xxxx-xxxx-xxxx-xxxx",help="The YOUTUBE STREAM RTMP Key")
    parser.add_argument("--YOUTUBE_STREAM_RES",type=str,default='720p',help="Youtube Stream Height")
    parser.add_argument("--RTSP_PATH",type=str,default="xxxx-xxxx-xxxx-xxxx-xxxx",help="The RTSP Path")
    parser.add_argument("--RTSP_SERVER_PATH",type=str,default="/media/steven/Elements/Full_Loop_YOLO/resources/rtsp_server.py",help="The path to rtsp_server.py")
    parser.add_argument("--fps",default=30,help="fps of incoming images for rtsp_server",type=int)
    parser.add_argument("--width",default=None,help="width of incoming images for rtsp_server",type=int)
    parser.add_argument("--height",default=None,help="height of incoming images for rtsp_server",type=int)
    parser.add_argument("--port",default=8554,help="port for rtsp_server",type=int)
    parser.add_argument("--stream_key",default="/video_stream",help="rtsp image stream uri for rtsp_server")
    parser.add_argument("--send_image_to_cell",action='store_true',help='Should send text messages with chips?')
    parser.add_argument("--send_image_to_cell_path",default="/media/steven/Elements/Full_Loop_YOLO/resources/send_image_to_cell.py",help="Send text message images of chips to cell")
    parser.add_argument("--destinations",type=str,default='XXXYYYZZZZ@mms.att.net',help='phone numbers to send text message updates to')
    parser.add_argument("--basepath_chips",type=str,default="/media/steven/Elements/chips",help="path for chips stored")
    parser.add_argument("--sleep_time_chips",type=float,default=30,help="Seconds to sleep between sending chips")
    opt = parser.parse_args()
    
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
