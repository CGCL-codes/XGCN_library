import XGCN
from XGCN.utils.parse_arguments import parse_arguments


def main():
    
    config = parse_arguments()

    model = XGCN.create_model(config)
    
    model.fit()


if __name__ == '__main__':
    
    main()
