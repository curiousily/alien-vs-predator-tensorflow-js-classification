import * as tf from "@tensorflow/tfjs";
import * as Plotly from "plotly.js-dist";
import * as tfvis from "@tensorflow/tfjs-vis";
import _ from "lodash";

var pixels = require("image-pixels");

const TENSOR_IMAGE_SIZE = 28;

const TRAIN_IMAGES_PER_CLASS = 347;
// const TRAIN_IMAGES_PER_CLASS = 50;
const VALID_IMAGES_PER_CLASS = 100;
// const VALID_IMAGES_PER_CLASS = 10;

const CLASSES = ["alien", "predator"];

const TRAIN_DIR_URL =
  "https://raw.githubusercontent.com/curiousily/tfjs-data/master/avp/train/";

const VALID_DIR_URL =
  "https://raw.githubusercontent.com/curiousily/tfjs-data/master/avp/validation/";

const loadImages = async urls => {
  const imageData = await pixels.all(urls, { clip: [0, 0, 192, 192] });
  return imageData.map(d =>
    rescaleImage(resizeImage(toSquare(tf.browser.fromPixels(d))))
  );
};

const toSquare = img => {
  const width = img.shape[0];
  const height = img.shape[1];

  // use the shorter side as the size to which we will crop
  const shorterSide = Math.min(img.shape[0], img.shape[1]);

  // calculate beginning and ending crop points
  const startingHeight = (height - shorterSide) / 2;
  const startingWidth = (width - shorterSide) / 2;
  const endingHeight = startingHeight + shorterSide;
  const endingWidth = startingWidth + shorterSide;

  // return image data cropped to those points
  return img.slice(
    [startingWidth, startingHeight, 0],
    [endingWidth, endingHeight, 3]
  );
};

const resizeImage = image =>
  tf.image.resizeBilinear(image, [TENSOR_IMAGE_SIZE, TENSOR_IMAGE_SIZE]);

const rescaleImage = image => {
  const batchedImage = image.expandDims(0);

  return batchedImage
    .toFloat()
    .div(tf.scalar(127))
    .sub(tf.scalar(1));
  // .sub(tf.scalar(1));
};

const buildModel = () => {
  const {
    conv2d,
    maxPooling2d,
    flatten,
    dense,
    batchNormalization,
    dropout,
    activation
  } = tf.layers;

  const model = tf.sequential();

  model.add(
    conv2d({
      inputShape: [TENSOR_IMAGE_SIZE, TENSOR_IMAGE_SIZE, 3],
      kernelSize: 3,
      filters: 32,
      activation: "relu"
    })
  );
  model.add(maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));
  model.add(maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));
  model.add(maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(flatten());
  model.add(dense({ units: 128, activation: "relu" }));
  model.add(dropout({ rate: 0.2 }));
  model.add(dense({ units: 64, activation: "relu" }));
  model.add(dropout({ rate: 0.2 }));
  model.add(dense({ units: 1, activation: "sigmoid" }));

  // model.add(
  //   conv2d({
  //     inputShape: [TENSOR_IMAGE_SIZE, TENSOR_IMAGE_SIZE, 3],
  //     kernelSize: 3,
  //     filters: 16,
  //     activation: "relu"
  //   })
  // );
  // model.add(maxPooling2d({ poolSize: 2, strides: 2 }));
  // model.add(conv2d({ kernelSize: 3, filters: 32 }));
  // model.add(batchNormalization());
  // model.add(activation({ activation: "relu" }));
  // model.add(maxPooling2d({ poolSize: 2, strides: 2 }));
  // // model.add(conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));
  // model.add(conv2d({ kernelSize: 3, filters: 32 }));
  // model.add(batchNormalization());
  // model.add(activation({ activation: "relu" }));

  // model.add(flatten());
  // model.add(dense({ units: 64, activation: "relu" }));
  // model.add(dropout({ relu: 0.2 }));
  // model.add(dense({ units: 2, activation: "softmax" }));
  model.compile({
    optimizer: tf.train.adam(0.0001),
    // optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });
  return model;
};

const rescale = (tensor, lowerBound, upperBound) => {
  const min = tf.min(tensor);
  const max = tf.max(tensor);

  const scaled = tensor.sub(min).div(max.sub(min));
  return scaled.mul(upperBound - lowerBound).add(lowerBound);
};

const showImageFromTensor = async tensor => {
  const canvasPreview = document.getElementById("tensor-image-preview");
  canvasPreview.width = TENSOR_IMAGE_SIZE;
  canvasPreview.height = TENSOR_IMAGE_SIZE;

  await tf.browser.toPixels(
    rescale(tf.squeeze(tensor), 0.0, 1.0),
    canvasPreview
  );
};

const run = async () => {
  console.log("Hello");

  const trainAlienImagePaths = _.range(0, TRAIN_IMAGES_PER_CLASS).map(
    num => `${TRAIN_DIR_URL}alien/${num}.jpg`
  );

  const trainPredatorImagePaths = _.range(0, TRAIN_IMAGES_PER_CLASS).map(
    num => `${TRAIN_DIR_URL}predator/${num}.jpg`
  );

  const validAlienImagePaths = _.range(0, VALID_IMAGES_PER_CLASS).map(
    num => `${VALID_DIR_URL}alien/${num}.jpg`
  );

  const validPredatorImagePaths = _.range(0, VALID_IMAGES_PER_CLASS).map(
    num => `${VALID_DIR_URL}predator/${num}.jpg`
  );

  const trainAlienImages = await loadImages(trainAlienImagePaths);
  const trainPredatorImages = await loadImages(trainPredatorImagePaths);

  const testAlienImages = await loadImages(validAlienImagePaths);
  const testPredatorImages = await loadImages(validPredatorImagePaths);

  try {
    const res = tf.layers
      .conv2d({ kernelSize: 3, filters: 3 })
      .apply(trainPredatorImages[0]);

    // await showImageFromTensor(res);

    // await showImageFromTensor(trainAlienImages[0]);

    const trainAlienTensors = tf.concat(trainAlienImages);

    const trainPredatorTensors = tf.concat(trainPredatorImages);

    const trainAlienLabels = tf.tensor1d(
      _.times(TRAIN_IMAGES_PER_CLASS, _.constant(0)),
      "int32"
    );

    const trainPredatorLabels = tf.tensor1d(
      _.times(TRAIN_IMAGES_PER_CLASS, _.constant(1)),
      "int32"
    );

    const testAlienTensors = tf.concat(testAlienImages);

    const testPredatorTensors = tf.concat(testPredatorImages);

    const testAlienLabels = tf.tensor1d(
      _.times(VALID_IMAGES_PER_CLASS, _.constant(0)),
      "int32"
    );

    const testPredatorLabels = tf.tensor1d(
      _.times(VALID_IMAGES_PER_CLASS, _.constant(1)),
      "int32"
    );

    const xTrain = tf.concat([trainAlienTensors, trainPredatorTensors]);

    const yTrain = tf.concat([trainAlienLabels, trainPredatorLabels]);

    const xTest = tf.concat([testAlienTensors, testPredatorTensors]);

    const yTest = tf.concat([testAlienLabels, testPredatorLabels]);

    const model = buildModel();

    const lossContainer = document.getElementById("loss-cont");

    await model.fit(xTrain, yTrain, {
      batchSize: 32,
      validationSplit: 0.1,
      shuffle: true,
      epochs: 100,
      // epochs: 10,
      callbacks: tfvis.show.fitCallbacks(
        lossContainer,
        ["loss", "val_loss", "acc", "val_acc"],
        {
          callbacks: ["onEpochEnd"]
        }
      )
    });

    const preds = tf.squeeze(tf.round(model.predict(xTest)));
    const labels = yTest;
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = document.getElementById("confusion-matrix");
    tfvis.render.confusionMatrix(container, {
      values: confusionMatrix,
      tickLabels: ["Alien", "Predator"]
    });

    const predData = tf.concat([testAlienImages[1], testPredatorImages[1]]);

    const predResults = model.predict(predData).dataSync();

    const alienText = document.getElementById("alien-prediction");

    alienText.innerHTML = `Alien prediction: ${predResults[0]}`;

    const predatorText = document.getElementById("predator-prediction");

    predatorText.innerHTML = `Predator prediction: ${1.0 - predResults[1]}`;
  } catch (e) {
    console.log(e.message);
  }

  console.log("Blaster");
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
