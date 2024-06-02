import logging
import datetime
from utils import *
from models import *
from data_loader import *
from config import Args
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn

def train(args, student_model, teacher_model, train_dataset, valid_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, num_workers=8, shuffle=True, pin_memory=True)
    scaler = GradScaler(enabled=args.use_amp)
    distill_loss_fn = nn.MSELoss()
    cl_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.train_epochs):
        for data in train_dataloader:
            input_ids, attention_mask, labels = data
            with autocast():
                student_outputs = student_model(input_ids, attention_mask)
                teacher_outputs = teacher_model(input_ids, attention_mask)
                distill_loss = distill_loss_fn(student_outputs, teacher_outputs)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            break
        break

def evaluate(student_model):
    pass

if __name__ == '__main__':
    args = Args().get_parser()
    logger = logging.getLogger(__name__)
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m%d_%H%M%S")

    task_suffix = formatted_time
    for key_arg in args.key_args:
        task_suffix += f"_{key_arg}-{getattr(args, key_arg)}"

    logger.setLevel(logging.INFO)  # 设置日志级别，可以根据需要进行调整

    # 创建一个处理程序，用于写入日志文件
    file_handler = logging.FileHandler(f'{task_suffix}.log')
    file_handler.setLevel(logging.INFO)  # 设置日志级别，可以根据需要进行调整

    # 创建一个处理程序，用于将日志输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # 设置日志级别，可以根据需要进行调整

    # 创建一个格式器，定义日志的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 设置处理程序的格式器
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 为日志记录器添加处理程序
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    teacher_model = AutoModel.from_pretrained(args.teacher_model_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)


    student_model = AutoModel.from_pretrained(args.student_model_name)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)

    MSMarco = load_dataset('ms_marco', 'v2.1')
    train_dataset, valid_dataset, test_dataset = MSMarco['train'], MSMarco['validation'], MSMarco['test']

    train(args, student_model, teacher_model, train_dataset, valid_dataset)