import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import _ from "lodash";

import { renderFilters, renderLayer } from "./visualization";

var pixels = require("image-pixels");

const TENSOR_IMAGE_SIZE = 28;

const TRAIN_IMAGES_PER_CLASS = 347;
// const TRAIN_IMAGES_PER_CLASS = 10;
const VALID_IMAGES_PER_CLASS = 100;
// const VALID_IMAGES_PER_CLASS = 5;

const TRAIN_DIR_URL =
  "https://raw.githubusercontent.com/curiousily/tfjs-data/master/avp/train/";

const VALID_DIR_URL =
  "https://raw.githubusercontent.com/curiousily/tfjs-data/master/avp/validation/";

const loadImages = async urls => {
  const imageData = await pixels.all(urls, { clip: [0, 0, 192, 192] });
  return imageData.map(d =>
    rescaleImage(resizeImage(toGreyScale(tf.browser.fromPixels(d))))
  );
};

const toGreyScale = image => image.mean(2).expandDims(2);

const resizeImage = image =>
  tf.image.resizeBilinear(image, [TENSOR_IMAGE_SIZE, TENSOR_IMAGE_SIZE]);

const rescaleImage = image => {
  const batchedImage = image.expandDims(0);

  return batchedImage
    .toFloat()
    .div(tf.scalar(127))
    .sub(tf.scalar(1));
};

const buildModel = () => {
  const { conv2d, maxPooling2d, flatten, dense } = tf.layers;

  const model = tf.sequential();

  model.add(
    conv2d({
      name: "first-conv-layer",
      inputShape: [TENSOR_IMAGE_SIZE, TENSOR_IMAGE_SIZE, 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling"
    })
  );
  model.add(maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(
    conv2d({
      name: "second-conv-layer",
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling"
    })
  );
  model.add(maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(flatten());

  model.add(
    dense({
      units: 1,
      kernelInitializer: "varianceScaling",
      activation: "sigmoid"
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.0001),
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
  const canvasEl = document.getElementById("tensor-image-preview");
  canvasEl.width = TENSOR_IMAGE_SIZE;
  canvasEl.height = TENSOR_IMAGE_SIZE;

  await tf.browser.toPixels(rescale(tf.squeeze(tensor), 0.0, 1.0), canvasEl);
};

const run = async () => {
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

  try {
    const trainAlienImages = await loadImages(trainAlienImagePaths);
    const trainPredatorImages = await loadImages(trainPredatorImagePaths);

    const testAlienImages = await loadImages(validAlienImagePaths);
    const testPredatorImages = await loadImages(validPredatorImagePaths);

    const res = tf.layers
      .conv2d({ kernelSize: 3, filters: 3 })
      .apply(trainPredatorImages[0]);

    // await showImageFromTensor(res);

    // await showImageFromTensor(trainPredatorImages[0]);

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
      epochs: 150,
      callbacks: tfvis.show.fitCallbacks(
        lossContainer,
        ["loss", "val_loss", "acc", "val_acc"],
        {
          callbacks: ["onEpochEnd"]
        }
      )
    });

    await model.save("localstorage://cnn-model");

    // const model = await tf.loadLayersModel("localstorage://cnn-model");

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

    const activationExamples = tf.concat([
      trainAlienTensors.slice(0, 5),
      trainPredatorTensors.slice(0, 5)
    ]);

    renderLayer(model, "first-conv-layer", "first-layer-container");

    renderFilters(
      model,
      activationExamples,
      "first-conv-layer",
      "first-layer-filters",
      TENSOR_IMAGE_SIZE
    );

    renderLayer(model, "second-conv-layer", "second-layer-container");

    renderFilters(
      model,
      activationExamples,
      "second-conv-layer",
      "second-layer-filters",
      TENSOR_IMAGE_SIZE
    );
  } catch (e) {
    console.log(e.message);
  }
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
