import * as tf from "@tensorflow/tfjs";
import * as Plotly from "plotly.js-dist";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as d3 from "d3";
import _ from "lodash";

var pixels = require("image-pixels");

const TENSOR_IMAGE_SIZE = 28;

const TRAIN_IMAGES_PER_CLASS = 347;
// const TRAIN_IMAGES_PER_CLASS = 10;
const VALID_IMAGES_PER_CLASS = 100;
// const VALID_IMAGES_PER_CLASS = 5;

const CLASSES = ["alien", "predator"];

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

const getActivation = (input, model, layer) => {
  const activationModel = tf.model({
    inputs: model.input,
    outputs: layer.output
  });
  return activationModel.predict(input);
};

const renderImage = async (container, tensor, imageOpts) => {
  const resized = tf.tidy(() =>
    tf.image
      .resizeNearestNeighbor(tensor, [imageOpts.height, imageOpts.width])
      .clipByValue(0.0, 1.0)
  );
  const canvas = document.createElement("canvas");
  canvas.width = imageOpts.width;
  canvas.height = imageOpts.height;
  canvas.style = `margin: 4px; width:${imageOpts.width}px; height:${
    imageOpts.height
  }px`;
  container.appendChild(canvas);
  await tf.browser.toPixels(resized, canvas);
  resized.dispose();
};

const renderImageTable = (container, headerData, data) => {
  let table = d3.select(container).select("table");

  if (table.size() === 0) {
    table = d3.select(container).append("table");
    table.append("thead").append("tr");
    table.append("tbody");
  }

  const headers = table
    .select("thead")
    .select("tr")
    .selectAll("th")
    .data(headerData);
  const headersEnter = headers.enter().append("th");
  headers.merge(headersEnter).each((d, i, group) => {
    const node = group[i];

    if (typeof d === "string") {
      node.innerHTML = d;
    } else {
      renderImage(node, d, {
        width: 25,
        height: 25
      });
    }
  });
  const rows = table
    .select("tbody")
    .selectAll("tr")
    .data(data);
  const rowsEnter = rows.enter().append("tr");
  const cells = rows
    .merge(rowsEnter)
    .selectAll("td")
    .data(d => d);
  const cellsEnter = cells.enter().append("td");
  cells.merge(cellsEnter).each((d, i, group) => {
    const node = group[i];
    renderImage(node, d, {
      width: 40,
      height: 40
    });
  });
  cells.exit().remove();
  rows.exit().remove();
};

const getActivationTable = (model, xTrain, layerName) => {
  const layer = model.getLayer(layerName);

  let filters = tf.tidy(() =>
    layer.kernel.val.transpose([3, 0, 1, 2]).unstack()
  );

  if (filters[0].shape[2] > 3) {
    filters = filters.map((d, i) => `Filter ${i + 1}`);
  }

  filters.unshift("Input");

  const activations = tf.tidy(() => {
    return getActivation(xTrain, model, layer).unstack();
  });
  const activationImageSize = activations[0].shape[0]; // e.g. 24

  const numFilters = activations[0].shape[2]; // e.g. 8

  const filterActivations = activations.map((activation, i) => {
    const unpackedActivations = Array(numFilters)
      .fill(0)
      .map((_, i) =>
        activation.slice(
          [0, 0, i],
          [activationImageSize, activationImageSize, 1]
        )
      );

    const inputExample = tf.tidy(() =>
      xTrain.slice([i], [1]).reshape([TENSOR_IMAGE_SIZE, TENSOR_IMAGE_SIZE, 1])
    );
    unpackedActivations.unshift(inputExample);
    return unpackedActivations;
  });
  return {
    filters,
    filterActivations
  };
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
      // epochs: 2,
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

    // tfvis.show.layer(
    //   document.getElementById("conv-layer-container"),
    //   model.getLayer("first-conv-layer")
    // );

    // const { filters, filterActivations } = getActivationTable(
    //   model,
    //   xTrain,
    //   "first-conv-layer"
    // );
    // renderImageTable(
    //   document.getElementById("conv-layer-filters"),
    //   filters,
    //   filterActivations
    // );
  } catch (e) {
    console.log(e.message);
  }
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
