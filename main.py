from train.train import train, test
from config import Config  

if __name__ == '__main__':
    if Config.mode == 'train':
        train()
    elif Config.mode == 'test':
        test()
    else:
        print("Invalid mode. Please set mode to 'train' or 'test' in the config.")
