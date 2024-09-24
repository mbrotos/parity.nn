import random
import argparse
import csv
import os

random.seed(42)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for bit-string parity prediction"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--bit_length", type=int, default=10, help="Length of the bit string"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data.csv",
        help="Output file to store the generated data",
    )
    return parser.parse_args(args)


def generate_bit_string(length):
    return [random.randint(0, 1) for _ in range(length)]


def generate_parity(bit_string):
    return sum(bit_string) % 2


def generate_data(num_samples, bit_length):
    return [
        (
            generate_bit_string(bit_length),
            generate_parity(generate_bit_string(bit_length)),
        )
        for _ in range(num_samples)
    ]


def save_data(data, output_file):
    out_path = os.path.join("data", output_file)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bit_string", "parity"])
        for bit_string, parity in data:
            writer.writerow(["".join(map(str, bit_string)), parity])


def main():
    args = parse_args()
    data = generate_data(args.num_samples, args.bit_length)
    save_data(data, args.output_file)


if __name__ == "__main__":
    main()
