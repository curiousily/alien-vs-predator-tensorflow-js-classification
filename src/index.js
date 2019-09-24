import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as Plotly from "plotly.js-dist";
import _ from "lodash";

const run = async () => {};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
