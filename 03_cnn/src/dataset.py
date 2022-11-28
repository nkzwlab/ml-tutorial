import torchvision

def get_mnist_dataset():
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ]
    )
    train_dataset = torchvision.datasets.MNIST(root = 'data', train = True, download = True, transform = trans)
    test_dataset = torchvision.datasets.MNIST(root = 'data', train = False, download = True, transform = trans)

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = get_mnist_dataset()

    print(train_dataset)
    print('========================')
    print(test_dataset)