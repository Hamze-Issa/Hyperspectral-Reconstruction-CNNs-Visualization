from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py

def paint_around(im, y, x, d=10, outer=100):
    r = int(d/2)
    painted = im.copy()
    painted[y-r:y+r, x-r:x+r, :] = im[y, x, :]
    return painted

def black_around_painted(im, y, x, inner=10, outer=200):
    inner = int(inner/2)
    outer = int(outer/2)
    painted = im.copy()
    painted[y-outer:y+outer, x-outer:x+outer, :] = np.array([0, 0, 0])
    painted[y, x, :] = im[y, x, :]
    painted[y-inner:y+inner, x-inner:x+inner, :] = im[y, x, :]
    return painted

def black_around(im, y, x, inner=10, outer=200):
    inner = int(inner/2)
    outer = int(outer/2)
    painted = im.copy()
    painted[y-outer:y+outer, x-outer:x+outer, :] = np.array([0, 0, 0])
    painted[y-inner:y+inner, x-inner:x+inner, :] = im[y-inner:y+inner, x-inner:x+inner, :]
    return painted


class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] ==bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)

class ValidDataset_2(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            for w in range(20):
                hyper_path = hyper_data_path + hyper_list[i]
                if 'mat' not in hyper_path:
                    continue
                with h5py.File(hyper_path, 'r') as mat:
                    hyper = np.float32(np.array(mat['cube']))

                ######################################################
                line_out = hyper[:, 35-w : 35, :].copy()
                line_in = hyper[:, 500-w : 500, :].copy()
                hyper[:, 35-w : 35, :] = line_in
                hyper[:, 500-w : 500, :] = line_out
                ######################################################

                hyper = np.transpose(hyper, [0, 2, 1])
                bgr_path = bgr_data_path + bgr_list[i]
                assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
                bgr = cv2.imread(bgr_path)

                ######################################################
                line_out = bgr[:, 35-w : 35, :].copy()
                line_in = bgr[:, 500-w : 500, :].copy()
                bgr[:, 35-w : 35, :] = line_in
                bgr[:, 500-w : 500, :] = line_out
                ######################################################

                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr)
                bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
                bgr = np.transpose(bgr, [2, 0, 1])
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                mat.close()
                print(f'Ntire2022 scene {w} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)

class ValidDataset_3(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [2, 0, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)

class ValidDataset_4(Dataset):
    

    def __init__(self, data_root, bgr2rgb=True):
        ######################################################
        outer = 50
        pixel_mat = [[100, -100], [100, 100], [-100, 100], [-100, -100]]
        ######################################################
        self.hypers = []
        self.bgrs = []
        self.indicators = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))

            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)

            ######################################################
            print(bgr.shape)
            for p in pixel_mat:
                x, y = p[0], p[1]
                for k, d in enumerate(range(1, 50, 5)):
                    bgr2 = bgr.copy()
                    bgr2  = paint_around(bgr2, y, x, d, outer)
            ######################################################
                    ####INDENTEDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD#############
                    if bgr2rgb:
                        bgr2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
                    bgr2 = np.float32(bgr2)
                    bgr2 = (bgr2 - bgr2.min()) / (bgr2.max() - bgr2.min())
                    bgr2 = np.transpose(bgr2, [2, 0, 1])
                    self.hypers.append(hyper)
                    self.bgrs.append(bgr2)
                    self.indicators.append([i, d, p])
                    mat.close()
                    print(f'Ntire2022 scene {i}, size {d}, pixel {p} is loaded.')
                    ####INDENTEDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD#############

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)
    
    def __getindicator__(self, idx):
        indicator = self.indicators[idx]
        return indicator
