import XGCN
from XGCN.utils.parse_arguments import parse_arguments


def main():
    
    config = parse_arguments()

    model = XGCN.train_model(config)


if __name__ == '__main__':
    
    main()
