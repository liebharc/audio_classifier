const ort = require('onnxruntime-node'); // or: onnxruntime-web
const { createCanvas, loadImage } = require('canvas');
const { Tensor } = ort;

const categories = ['music', 'noise', 'speech'];
const modelPath = 'audio_classifier.onnx';

async function loadModel() {
	return await ort.InferenceSession.create(modelPath);
}

async function preprocessImage(imagePath, startX, width, height) {
	const image = await loadImage(imagePath);
	const canvas = createCanvas(width, height);
	const ctx = canvas.getContext('2d');
	ctx.drawImage(image, startX, 0, width, height, 0, 0, width, height);

	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
	const floatData = new Float32Array(3 * width * height);

	for (let i = 0; i < width * height; i++) {
		floatData[i] = imageData[i * 4] / 255.0; // Red channel
		floatData[i + width * height] = imageData[i * 4 + 1] / 255.0; // Green channel
		floatData[i + 2 * width * height] = imageData[i * 4 + 2] / 255.0; // Blue channel
	}

	return new Tensor('float32', floatData, [1, 3, height, width]);
}

async function predict(model, imagePath) {
	const image = await loadImage(imagePath);
	const windowSize = 28;
	const stepSize = windowSize / 2;
	const height = image.height;
	const width = image.width;

	for (let startX = 0; startX + windowSize <= width; startX += stepSize) {
		const inputTensor = await preprocessImage(
			imagePath,
			startX,
			windowSize,
			height
		);
		const results = await model.run({ input: inputTensor });
		const output = results.output.data;
		const predictionIndex = output.indexOf(Math.max(...output));
		console.log(`${startX}: ${categories[predictionIndex]}`);
	}
}

(async () => {
	const model = await loadModel();
	const imageFile = process.argv[2];
	if (!imageFile) {
		console.log('Please provide an image file path.');
		process.exit(1);
	}
	await predict(model, imageFile);
})();
