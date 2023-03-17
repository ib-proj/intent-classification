import argparse

from engine import engine


def main():
    parser = argparse.ArgumentParser(description='Run the SWDA classification engine.')
    parser.add_argument('--model_name', type=str, default='GruEncoder', help='Name of the model to use')
    args = parser.parse_args()

    engine(model_name=args.model_name)


if __name__ == "__main__":
    main()
