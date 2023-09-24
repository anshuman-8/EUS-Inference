import logging as log


def main():

    inference_config = {
        'checkpoint_path': 'checkpoints/epoch=9.ckpt',
        'device': 'cpu',
        'image_size': 224,
        'mean': [0.22782720625400543, 0.22887665033340454, 0.23145385086536407],
        'std': [0.11017259210348129, 0.11015155166387558, 0.11037711054086685],
        'station_class': {0: "Station 1", 1: "Station 2", 2: "Station 3"},
        'log_level': log.INFO,
        'log_format': '%(asctime)s %(levelname)s %(message)s',
        'video_source': 2,
        'fps': 30,
    }

    print("Hello World!")


if __name__ == "__main__":

    
    main()
