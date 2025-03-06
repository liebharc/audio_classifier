const fs = require('fs');
const { WaveFile } = require('wavefile');
const FFT = require('fft.js');
const { createCanvas } = require('canvas');

class MelFilterBank {
	constructor(samplerate, nfft, lowFreq, highFreq, nMel) {
		this.samplerate = samplerate;
		this.nfft = nfft;
		this.lowFreq = lowFreq;
		this.highFreq = highFreq;
		this.nMel = nMel;

		this.lowMel = this.fToMel(lowFreq);
		this.highMel = this.fToMel(highFreq);
		this.dMel = (this.highMel - this.lowMel) / (nMel + 1);

		this.melPoints = this.calculateMelPoints();
		this.freqBins = this.melPoints.map((m) => this.melToF(m));
		this.binIndices = this.freqBins.map((f) =>
			Math.floor(((nfft + 1) * f) / samplerate)
		);
		this.filters = this.createFilterBanks();
	}

	fToMel(f) {
		return 2595 * Math.log10(1 + f / 700);
	}

	melToF(m) {
		return 700 * (Math.pow(10, m / 2595) - 1);
	}

	calculateMelPoints() {
		return Array.from(
			{ length: this.nMel + 2 },
			(_, i) => this.lowMel + i * this.dMel
		);
	}

	createFilterBanks() {
		const filters = [];
		for (let m = 1; m <= this.nMel; m++) {
			let filter = new Array(this.nfft / 2 + 1).fill(0);
			for (let k = 0; k < this.nfft / 2 + 1; k++) {
				if (k < this.binIndices[m - 1]) continue;
				if (k < this.binIndices[m]) {
					filter[k] =
						(k - this.binIndices[m - 1]) /
						(this.binIndices[m] - this.binIndices[m - 1]);
				} else if (k <= this.binIndices[m + 1]) {
					filter[k] =
						(this.binIndices[m + 1] - k) /
						(this.binIndices[m + 1] - this.binIndices[m]);
				}
			}
			filters.push(filter);
		}
		return filters;
	}

	applyFilterBank(spectrum) {
		return this.filters.map((filter) =>
			filter.reduce((sum, weight, idx) => sum + weight * spectrum[idx], 0)
		);
	}
}

class MelSpectrogram {
	constructor(samplerate, frameSize, nMel, minFreq, maxFreq) {
		this.fft = new FFT(frameSize);
		this.nfft = frameSize;
		this.filterBank = new MelFilterBank(
			samplerate,
			frameSize,
			minFreq,
			maxFreq,
			nMel
		);
	}

	compute(signal) {
		const windowed = this.applyHammingWindow(signal);
		const spectrum = this.computeFFT(windowed);
		return this.filterBank.applyFilterBank(spectrum);
	}

	applyHammingWindow(buffer) {
		return buffer.map(
			(val, idx) =>
				val *
				(0.54 - 0.46 * Math.cos((2 * Math.PI * idx) / (buffer.length - 1)))
		);
	}

	computeFFT(buffer) {
		const out = this.fft.createComplexArray();
		this.fft.realTransform(out, buffer);
		this.fft.completeSpectrum(out);
		const spectrum = new Array(this.nfft / 2 + 1);
		for (let i = 0; i < spectrum.length; i++) {
			const real = out[2 * i];
			const imag = out[2 * i + 1];
			spectrum[i] = Math.sqrt(real * real + imag * imag);
		}
		return spectrum;
	}
}

function processWavFile(filePath) {
	const data = fs.readFileSync(filePath);
	const spectrogramData = processBuffer(data);

	saveSpectrogram(spectrogramData, 'debug_spectrogram_js.png');
}

function processBuffer(
	data,
	frameSize = 2048,
	hopSize = 512,
	nMel = 40,
	minFreq = 300,
	maxFreq = 8000
) {
	const wav = new WaveFile(data);
	wav.toBitDepth('32f');
	const audioSamples = new Float32Array(wav.getSamples(false)[0]);
	const sampleRate = wav.fmt.sampleRate;

	const melSpec = new MelSpectrogram(
		sampleRate,
		frameSize,
		nMel,
		minFreq,
		maxFreq
	);
	const spectrogramData = [];

	const maxSampleValue = 32 * 1024;

	for (let i = 0; i + frameSize <= audioSamples.length; i += hopSize) {
		const chunk = audioSamples
			.slice(i, i + frameSize)
			.map((v) => v * maxSampleValue);
		const melData = melSpec.compute(chunk);
		spectrogramData.push(melData);
	}

	return spectrogramData;
}

function drawSpectrogram(spectrogramData) {
	const width = spectrogramData.length;
	const height = spectrogramData[0].length;
	const canvas = createCanvas(width, height);
	const ctx = canvas.getContext('2d');

	let minValue = spectrogramData[0][0];
	let maxValue = spectrogramData[0][0];

	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			const value = spectrogramData[x][y];
			if (value < minValue) {
				minValue = value;
			}
			if (value > maxValue) {
				maxValue = value;
			}
		}
	}

	const imageData = ctx.getImageData(0, 0, width, height);
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			const value = Math.floor(
				((spectrogramData[x][y] - minValue) / (maxValue - minValue)) * 255
			);
			const index = ((height - y - 1) * width + x) * 4;
			imageData.data[index] = value;
			imageData.data[index + 1] = value;
			imageData.data[index + 2] = value;
			imageData.data[index + 3] = 255;
		}
	}
	ctx.putImageData(imageData, 0, 0);
	return canvas;
}

function saveSpectrogram(spectrogramData, filename) {
	const canvas = drawSpectrogram(spectrogramData);
	fs.writeFileSync(filename, canvas.toBuffer('image/png'));
}

// Example usage
processWavFile('your_audio_file.wav');
