# CNN Digit Recognizer

This repository contains a deep learning project that utilizes Convolutional Neural Networks (CNN) to build a digit recognizer using the MNIST dataset. The MNIST dataset consists of a large collection of handwritten digit images, commonly used as a benchmark in the field of computer vision.

## Dependencies

The project requires the following main packages to be installed:

- numpy
- matplotlib
- tensorflow
- keras

Please ensure you have these packages installed with the appropriate versions.

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/TheStrange-007/DigitRecognizer.git
```

2. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

3. Run the Flask server using the following command:

```bash
python server.py
```

4. Access the web interface by visiting `http://localhost:5000` in your web browser.

5. Draw a digit in the provided interface, and the trained model will predict the corresponding digit.

Alternatively, you can quickly try the deployed model at [https://cnn-digitrecognizer.thestrange-007.repl.co](https://cnn-digitrecognizer.thestrange-007.repl.co).

## Model Training

The repository includes code for training the CNN model using the MNIST dataset. The model architecture consists of convolutional layers, pooling layers, dropout layers, and dense layers. Augmentation techniques are also applied to generate augmented images for training.

## Contributing

Contributions are welcome! If you find any issues or want to enhance the project, feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

