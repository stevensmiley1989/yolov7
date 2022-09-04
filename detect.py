import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
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
import simplejpeg
import imagezmq
import traceback
import socket
from multiprocessing import Process
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
    # ADD ability to send IMGZMQ images to other receivers 
    sender_dic={}
    if os.path.exists(opt.multi_sender_imgzmq_PATH):
        import sys
        sys.path.append(os.path.dirname(opt.multi_sender_imgzmq_PATH))
        import multi_sender_imgzmq as msi
        print('LOADED MSI')
        PORT_LIST=msi.generate_PORT_LIST(opt.PORT_LIST_PATH)
        sender_dic=msi.create_senders(msi.IP_LIST,PORT_LIST,REQ_REP=opt.REP_REQ)

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    resize_for_YOLO_MODEL=opt.resize_for_YOLO_MODEL
    targets_of_interest_list=opt.target_of_interest_list.split(';')
    #edit sjs
    PORT=opt.PORT
    HOST=opt.HOST
    use_socket=opt.use_socket
    SAVE_RAWVIDEO=opt.SAVE_RAWVIDEO
    INFERENCE_TENSORFLOW_path=opt.INFERENCE_TENSORFLOW_path
    if INFERENCE_TENSORFLOW_path!='None' and os.path.exists(INFERENCE_TENSORFLOW_path):
        INFERENCE_TENSORFLOW_path=INFERENCE_TENSORFLOW_path.replace("'","").replace('"',"")
        classify_chips=True
        cmd_i="bash "+INFERENCE_TENSORFLOW_path
        Process(target=run_cmd,args=(cmd_i,)).start()
        f=open(INFERENCE_TENSORFLOW_path,'r')
        f_read=f.readlines()
        f.close()
        SETTINGS_PATH=[w for w in f_read if w.find("--SETTINGS_PATH=")!=-1]
        SETTINGS_PATH=SETTINGS_PATH[0].split('--SETTINGS_PATH=')[1].lstrip(' ').split(' ')[0].replace('\n','').replace("'","").replace('"',"")
        f=open(SETTINGS_PATH,'r')
        f_read=f.readlines()
        f.close()
        PORT_TF=[w.split("RT=")[1] for w in f_read if w.find('PORT=')!=-1]
        PORT_TF=int(PORT_TF[0].replace('\n','').replace(' ',''))
        HOST_TF=[w.split('ST=')[1] for w in f_read if w.find('HOST=')!=-1]
        HOST_TF=HOST_TF[0].replace('\n','').replace(' ','').replace('"',"").replace("'","")
    else:
        classify_chips=False
    if os.path.exists(INFERENCE_TENSORFLOW_path)==False:
        print('THIS PATH DOES NOT EXIST = {}'.format(INFERENCE_TENSORFLOW_path))


    if  use_socket:
        import socket
        print('using Socket for PORT=={} and HOST=={}'.format(PORT,HOST))
        try:
            sendstuff=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sendstuff.connect((HOST, PORT)) #edit sjs
        except:
            print('Not accepting socket')
            use_socket=False
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
    print('clearing labels if they exist')
    os.system('rm -rf {}'.format(save_dir/'labels'))
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
    vid_path_raw,vid_writer_raw=None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride,resize=resize_for_YOLO_MODEL)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride,resize=resize_for_YOLO_MODEL)

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

            im0_og=im0.copy()
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            save_path_raw = str(save_dir / p.name)  # img.jpg
            save_path_raw=save_path_raw+'_raw'

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
                img_list={}
                label_list={}
                detection_time_i=str(time.time()).replace('.','point')
                detection_path_i=os.path.join(basepath_chips,detection_time_i)
                detection_path_i_text=os.path.join(detection_path_i,'message_content.txt')
                datetime_i=str(datetime.datetime.now())
                detection_path_i_full=os.path.join(detection_path_i,'FULL')
                im0_og=im0.copy()
                msg_i_list="&"
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


                    
                    if send_image_to_cell and os.path.exists(send_image_to_cell_path) and send_allowed and names[int(cls)] in targets_of_interest_list:
                        if os.path.exists(detection_path_i)==False:
                            os.makedirs(detection_path_i)
                        label = f'{names[int(cls)]} {conf:.2f}'
                        label_og=label
                        
                        label=label.replace(' ','_').replace('.','p').replace(':',"c").replace('-','_')
                        chip_i=label+".jpg"
                        chip_i=os.path.join(detection_path_i,chip_i)
                        label_list[chip_i]=label_og
                        boxes=xyxy
                        MARGIN=5
                        #print(im0_og.shape)
                        xmin=int(boxes[0].cpu().detach().numpy())
                        xmin=max(xmin-MARGIN,0)
                        xmax=int(boxes[2].cpu().detach().numpy())
                        xmax=min(xmax+MARGIN,im0_og.shape[1])
                        ymin=int(boxes[1].cpu().detach().numpy())
                        ymin=max(ymin-MARGIN,0)
                        ymax=int(boxes[3].cpu().detach().numpy())
                        ymax=min(ymax+MARGIN,im0_og.shape[0])
                        #print('xmin,xmax,ymin,ymax')
                        #print(xmin,xmax,ymin,ymax)
                        if len(list(im0_og.shape))==3:
                            img_list[chip_i]=im0_og[ymin:ymax,xmin:xmax,:]
                            cv2.imwrite(chip_i,im0_og[ymin:ymax,xmin:xmax,:])
                        elif len(list(im0_og.shape))==2:
                            img_list[chip_i]=im0_og[ymin:ymax,xmin:xmax]
                            cv2.imwrite(chip_i,im0_og[ymin:ymax,xmin:xmax])
                    if classify_chips:
                        boxes=xyxy
                        MARGIN=5
                        #print(im0_og.shape)
                        xmin=int(boxes[0].cpu().detach().numpy())
                        xmin=max(xmin-MARGIN,0)
                        xmax=int(boxes[2].cpu().detach().numpy())
                        xmax=min(xmax+MARGIN,im0_og.shape[1])
                        ymin=int(boxes[1].cpu().detach().numpy())
                        ymin=max(ymin-MARGIN,0)
                        ymax=int(boxes[3].cpu().detach().numpy())
                        ymax=min(ymax+MARGIN,im0_og.shape[0])
                        #print('xmin,xmax,ymin,ymax')
                        #print(xmin,xmax,ymin,ymax)
                        print(f'OBJECT DETECTION CHIP PREDICTION = {names[int(cls)]} {conf:.2f}')
                        currentCHIP=im0_og[ymin:ymax,xmin:xmax,:]
                        currentCHIP = np.ascontiguousarray(currentCHIP)
                        classification_i,confidence_i=classify_chip(currentCHIP,HOST_TF,PORT_TF,host_name)
                    if save_img or view_img:  # Add bbox to image
                        if classify_chips:
                            label=f'OBJ: {names[int(cls)]} {conf:.2f} CL: {classification_i} {confidence_i:.2f}'
                        else:
                            label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    if use_socket:
                        boxes=xyxy
                        MARGIN=0
                        #print(im0_og.shape)
                        xmin=int(boxes[0].cpu().detach().numpy())
                        xmin=max(xmin-MARGIN,0)
                        xmax=int(boxes[2].cpu().detach().numpy())
                        xmax=min(xmax+MARGIN,im0_og.shape[1])
                        ymin=int(boxes[1].cpu().detach().numpy())
                        ymin=max(ymin-MARGIN,0)
                        ymax=int(boxes[3].cpu().detach().numpy())
                        ymax=min(ymax+MARGIN,im0_og.shape[0])
                        prefix=opt.socket_prefix
                        msg_i=f'{prefix}_{names[int(cls)]};{xmin};{ymin};{xmax};{ymax};{conf};{im0.shape[1]};{im0.shape[0]}' #edit sjs
                        msg_i_list=msg_i_list+msg_i+"&"
                if use_socket:
                    #print('msg_i_list=',msg_i_list)
                    sendstuff.sendall(msg_i_list.encode())#edit sjs
                if send_image_to_cell and os.path.exists(send_image_to_cell_path) and send_allowed and len(img_list)>0:
                    if os.path.exists(detection_path_i_full)==False:
                        os.makedirs(detection_path_i_full)
                    #subject=",".join(w for w in img_list.keys())
                    main_message="Targets FOUND"
                    with open(detection_path_i_text,'w') as f:
                        f.writelines('Time found == {};\n'.format(datetime_i))
                        # for k,v in label_list.items():
                        #     f.writelines(';{}\n'.format(v))
                    cmd_i='python3 {} --destinations="{}" --main_message="{}"  --img_path="{}" '.format(send_image_to_cell_path,destinations,main_message,detection_path_i)
                    print(cmd_i)
                    Thread(target=run_cmd,args=(cmd_i,)).start()
                    cv2.imwrite(os.path.join(detection_path_i_full,'Full_Detected.jpg'),im0)
                    cv2.imwrite(os.path.join(detection_path_i_full,'Full_OG.jpg'),im0_og)
                    send_allowed=False


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
                            cap_type='video'
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w, h = im0.shape[1], im0.shape[0] #edit sjs
                            
                        else:  # stream
                            cap_type='stream'
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                            save_path_raw +='.mp4'
                        if SAVE_RAWVIDEO and cap_type=='stream':
                            vid_writer = cv2.VideoWriter(save_path_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        else:
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    if SAVE_RAWVIDEO and cap_type=='stream':
                         vid_writer.write(im0_og)
                    else:
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
            
            if len(sender_dic)>0:
                msi.run_multi_senders_custom(im0_og,sender_dic)


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    if use_socket:
        data = sendstuff.recv(1024) #edit sjs
        print('Received', repr(data)) #edit sjs
host_name= socket.gethostname() # send hostname with each image
def classify_chip(currentCHIP,HOST,PORT,host_name):
    jpeg_quality = 100                   # 0 to 100, higher is better quality
    try:
        with imagezmq.ImageSender(connect_to=f'tcp://{HOST}:{PORT}') as sender:
            #while True:                 # send images as a stream until Ctrl-C
                #image          = picam.read()
            jpg_buffer     = simplejpeg.encode_jpeg(currentCHIP, quality=jpeg_quality, 
                                                    colorspace='BGR')
            classification_i = sender.send_jpg(host_name, jpg_buffer)
            classification_i=str(classification_i).replace("b'","").replace("'","").replace('"',"")
            print('classification_i',classification_i)
            if str(classification_i).find(';')!=-1:
                classification_i_og=classification_i
                classification_i=str(classification_i_og).split(';')[0]
                confidence_i=float(str(classification_i_og).split(';')[1])
            return classification_i,confidence_i
    except (KeyboardInterrupt, SystemExit):
        pass                            # Ctrl-C was pressed to end program
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()                

if __name__ == '__main__':
    #TF_INFERENCE=r"/media/steven/OneTouch4tb/DATA_CLASSIFICATION/train_hemtthumveetank/CUSTOM_hemtthumveetank_chipW256_chipH256_classes3/INFERENCE.sh"

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
    parser.add_argument('--nosave', action='store_false', help='do not save images/videos') #edit sjs can change, but this makes it not store video
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
    parser.add_argument("--use_socket",action='store_false',help='use socket to send boxes and label?')
    parser.add_argument("--PORT",dest='PORT',type=int,default=8889,help='port like 8889 for sending boxes to')
    parser.add_argument("--HOST",dest='HOST',type=str,default='10.5.1.201',help='This is the main server ip address to send to')
    parser.add_argument("--SAVE_RAWVIDEO",action='store_false',help='save the raw video of incoming video')
    parser.add_argument("--socket_prefix",default='top',type=str,help='for encoding with socket message with bbox')
    parser.add_argument("--multi_sender_imgzmq_PATH",default="/media/steven/Elements/Full_Loop_YOLO/resources/multi_sender_imgzmq.py",help='Path to multi_sender_imgzmq')
    parser.add_argument("--REP_REQ",action='store_true',help='Response Required for imgzmq')
    parser.add_argument("--PORT_LIST_PATH",default="/media/steven/Elements/Full_Loop_YOLO/resources/PORT_LIST.txt",help="port list path",type=str)
    parser.add_argument("--resize_for_YOLO_MODEL",action='store_false',help='resize incoming images to match yolo model input size?')
    parser.add_argument("--target_of_interest_list",default="person;tractor;boat",type=str,help='Decide to send chips of this object if sending via cell')
    parser.add_argument("--INFERENCE_TENSORFLOW_path",default='None',help="path to use secondary classifier with tensorflow models bash file")
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
