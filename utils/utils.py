import os
import time
import sys
import random 
import torch
import subprocess
import numpy as np


class GPUGet:
    def __init__(self,
                 min_gpu_number,
                 time_interval, mode="single"):
        self.min_gpu_number = min_gpu_number
        self.time_interval = time_interval
        self.mode = mode

    def get_gpu_info(self):
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')[1:]
        gpu_dict = dict()
        for i in range(len(gpu_status) // 4):
            index = i * 4
            gpu_state = str(gpu_status[index].split('   ')[2].strip())
            gpu_power = int(gpu_status[index].split('   ')[-1].split('/')[0].split('W')[0].strip())
            gpu_memory = int(gpu_status[index + 1].split('/')[0].split('M')[0].strip())
            gpu_dict[i] = (gpu_state, gpu_power, gpu_memory)
        return gpu_dict

    def loop_monitor(self, gpu): 
        available_gpus = []
        while True:
            gpu_dict = self.get_gpu_info()
            for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
                if gpu_power <= 80 and gpu_memory <= 10000:  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090
                    gpu_str = f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n "
                    sys.stdout.write(gpu_str)
                    sys.stdout.flush()
                    if gpu is not None:
                        if str(i) in gpu:
                            available_gpus.append(i)
                    else:
                        available_gpus.append(i)
            if len(available_gpus) >= self.min_gpu_number:
                return available_gpus
            else:
                available_gpus = []
                time.sleep(self.time_interval)

    def run(self, cmd_parameter, cmd_command, gpu=None, mode="external"):
        available_gpus = self.loop_monitor(gpu)
        if self.mode == "single":
            gpu_list_str = str(available_gpus[0])     # 仅使用一个GPU
        else:
            gpu_list_str = ",".join(map(str, available_gpus))
        if mode == "external":
            # print("use gpu: ", gpu_list_str)
            # 构建终端命令
            # cmd_parameter = fr"""{cmd_parameter}
            #                   NUM_GPUS={len(available_gpus)} ; \ """  # 一定要有 `; \ `
            cmd_parameter = fr"""{cmd_parameter}                       
                            NUM_GPUS=1 ; \ """  # 一定要有 `; \ `
            cmd_command = fr"""CUDA_VISIBLE_DEVICES={gpu_list_str}\
                            {cmd_command}"""
            # command = fr"""{cmd_parameter} {cmd_command}"""
            command = fr"""{cmd_command}"""
            print(command)
            # send_email(title='GPU FREE!', content="run: " + command)
            os.system(command)
            print("current time is: ", time.ctime())
        else:
            return gpu_list_str

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        print ('CUDA is available')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename="all_state.pth"):
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    torch.save(model, "model.pt")


def load_checkpoint(model, optimizer, scheduler, filename="my_checkpoint.pth"):
    print(f"=> Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, scheduler, epoch, loss

def init_logger(task_name):
    def get_git_revision_hash() -> str:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    
    log_dir = f"./logs/{task_name}/archive/"
    candidate_save_files = ["../main.py", "../config.py", "../models.py"]
    # save files
    for file in candidate_save_files:
        os.system("cp {} {}".format(file, os.path.join(log_dir, file.split("/")[-1])))

class ALI:
    def __init__(self) -> None:
        from aligo import Aligo
        # from aligo import Auth

        # Auth._EMAIL_HOST = 'smtp.126.com'
        # Auth._EMAIL_PORT = '25'
        # Auth._EMAIL_USER = 'ruili200009@126.com'
        # Auth._EMAIL_PASSWORD = 'SCEFALPUWMALIDUF'

        self.ali = Aligo()

    @staticmethod
    def upload_file(self, file_path='Research', parent_file_id=None, file_name='./dataset.zip'):
        folder = self.ali.get_folder_by_path(file_path)
        print(f"folder: {folder.file_id}")
        # user = ali.get_user()  # 获取用户信息
        # print(user.user_name, user.nick_name, user.phone)  # 打印用户信息
        self.ali.upload_file(file_name, parent_file_id=folder.file_id)

    @staticmethod
    def download_file(self, file_id, file_name='codesearch_data.zip'):
        file = self.ali.get_file_by_path(file_name)
        self.ali.download_file(file=file)

class PLT:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot(self):
        import matplotlib.pyplot as plt
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['font.family'] = 'Times New Roman'

        rounds = [1, 2, 3]
        chatgpt = [51.7, 54.9, 55.3]
        gpt4 = [68.2, 73.3, 74.0]  
        claude2 = [60.4, 62.1, 62.5]
        palm = [46.0, 47.2, 47.4]

        plt.plot(rounds, chatgpt, marker='o', label='ChatGPT')
        plt.plot(rounds, gpt4, marker='o', label='GPT-4')
        plt.plot(rounds, claude2, marker='o', label='Claude-2')
        plt.plot(rounds, palm, marker='o', label='PaLM')

        plt.xlabel("Rounds")
        plt.ylabel("pass@K")
        plt.legend(loc="upper left")
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1)) 
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

        # save pic as pdf and svg
        plt.savefig("passAtK.pdf", format="pdf")
        plt.savefig("passAtK.svg", format="svg")

        plt.show()


# if __name__ == '__main__':
#     min_gpu_number = 1  # 最小GPU数量，多于这个数值才会开始执行训练任务。
#     time_interval = 0.1  # 监控GPU状态的频率，单位秒。
#     gpu_get = GPUGet(min_gpu_number, time_interval)
#     print(gpu_get.get_gpu_info())

#     cmd_parameter = r""""""  # 命令会使用到的参数，使用 `;` 连接。
#     cmd_command = r"""nohup bash runML.sh > MLlog_CS_2.log 2>&1 &"""
#     gpu_get.run(cmd_parameter, cmd_command)
#     # nohup python -u train.py > trainSH.log 2>&1 &
