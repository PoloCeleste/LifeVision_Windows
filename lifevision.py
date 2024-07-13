import pygame, torch, cv2, os, time
from tkinter import Tk, Message

from yolact import Yolact
import torch.backends.cudnn as cudnn
from utils import timer
from data import COLORS
from collections import defaultdict

from utils.functions import SavePath
from data import cfg, set_cfg#, set_dataset
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from layers.output_utils import postprocess, undo_image_transformation

color_cache = defaultdict(lambda: {})

Belt_con = False
Helmet_con = False
Shoes_con = False

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    global Belt_con
    global Helmet_con
    global Shoes_con

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0.20)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:20]
        
        if cfg.eval_mask_branch:
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(20, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < 0.20:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    if cfg.eval_mask_branch and num_dets_to_consider > 0:
        masks = masks[:num_dets_to_consider, :, :, None]

        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        inv_alph_masks = masks * (-mask_alpha) + 1
        
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    if num_dets_to_consider == 0:
        return img_numpy

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]
        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        _class = cfg.dataset.class_names[classes[j]]
        if(_class == 'Belt on' or _class == 'Belt off' or _class == 'Helmet on' or _class == 'Helmet off' or _class == 'Shoes on' or _class == 'Shoes off'):
            if(_class == 'Belt on' and score >= 0.6):
                Belt_con=True
            elif(_class == 'Belt off' and score >= 0.5):
                Belt_con=False
            if(_class == 'Helmet on' and score >= 0.6):
                Helmet_con=True
            elif(_class == 'Helmet off' and score >= 0.5):
                Helmet_con=False
            if(_class == 'Shoes on' and score >= 0.6):
                Shoes_con=True
            elif(_class == 'Shoes on' and score < 0.4):
                _class = 'Shoes off'
            elif(_class == 'Shoes off' and score >= 0.5):
                Shoes_con=False

            text_str = '%s: %.2f' % (_class, score)

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

            text_pt = (x1, y1 - 3)
            text_color = [255, 255, 255]

            cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)            
    
    return img_numpy

from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    def gather(self, outputs, output_device):
        return sum(outputs, [])

def evalvideo(net:Yolact, path='0', out_path:str=None):
    cudnn.benchmark = True
    video_multiframe = 4
    is_webcam = True

    cudnn.benchmark = True
    
    vid = cv2.VideoCapture(0)
    vid.set(3,1280)
    vid.set(4,720)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    num_frames = float('inf')

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print('Clean...', end=' ')
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        print('Done.')
        play_gearinfo()

    def get_next_frame(vid):
        frames = []
        for idx in range(video_multiframe):
            frame = vid.read()[1]
            frame = cv2.flip(frame, -1)
            frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    video_fps = 0

    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None:
                        cv2.namedWindow(path, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(path, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow(path, frame_buffer.get())
                    else:
                        out.write(frame_buffer.get())
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                            % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                if out_path is None and cv2.waitKey(1) == 27:   # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001
                
                if out_path is None:
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)
        except:
            import traceback
            traceback.print_exc()


    extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None: print('Press Escape to close.')
    max_time_end = time.time()+20
    try:
        Playtop("Wait.wav", 'Wait 20sec for shoot...', de=5000)
        while vid.isOpened() and running:
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None
            
            if not (vid_done and len(active_frames) == 0):
                for frame in active_frames:
                    _args =  [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
                
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                active_frames = [x for x in active_frames if x['idx'] > 0]

                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)
                
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence)-1})

                frame_times.add(time.time() - start_time)
                fps = video_multiframe / frame_times.get_avg()
            else:
                fps = 0
            
            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (fps, video_fps, frame_buffer.qsize())
            if not False:
                print('\r' + fps_str + '    ', end='')
            if time.time()>max_time_end:
                time.sleep(1)
                print('\nDone.')
                running=False
                
    except KeyboardInterrupt:
        print('\nStopping...')    
    cleanup_and_exit()
    

def evaluate(net:Yolact):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False

    evalvideo(net)

def Play(targ, dl=0):
    pygame.mixer.init()
    pygame.mixer.music.load('assets/'+targ)
    pygame.mixer.music.play()
    time.sleep(dl)

def Playtop(targ, tag='', siz=50, de=2000):
    pygame.mixer.init()
    pygame.mixer.music.load('assets/'+targ)
    pygame.mixer.music.play()
    top(tag, siz, de)

def play_gearinfo():
    global Belt_con
    global Helmet_con
    global Shoes_con
    if not Belt_con:
        Play("B.wav", 1.5)
    if not Helmet_con:
        Play("H.wav", 1.5)
    if not Shoes_con:
        Play("S.wav", 1.5)
    if not Belt_con or not Helmet_con or not Shoes_con:
        Play("N.wav", 3)
    if Belt_con and Helmet_con and Shoes_con:
        Play("ALL.wav", 3)
    if not Belt_con:
        cmd='java -jar Belt.jar'
        Play("Belt.wav")
        os.system(cmd)
    if not Helmet_con:
        cmd='java -jar Helmet.jar'
        Play("Helmet.wav")
        os.system(cmd)
    if not Shoes_con:
        cmd='java -jar Shoes.jar'
        Play("Shoes.wav")
        os.system(cmd)

def top(comment='De', siz=50, dely=2000):
    top = Tk()
    Message(top, text=comment, font=("times", siz, "bold"), width=1080, padx=100, pady=200).pack()
    top.after(dely, top.destroy)
    top.mainloop()

def main():
    global Belt_con
    global Helmet_con
    global Shoes_con
    
    trained_model = 'weights/yolact_resnet101_safety_33_200000.pth'

    model_path = SavePath.from_str(trained_model)
    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    set_cfg(config)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if torch.cuda.is_available():
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(trained_model)
        net.eval()
        print(' Done.') 
        
        if torch.cuda.is_available():
            net = net.cuda()

        Playtop("Backstep.wav", 'Please step back for the shoot.', de=5000)

        evaluate(net)
        return Belt_con, Helmet_con, Shoes_con

if __name__ == '__main__':
    main()