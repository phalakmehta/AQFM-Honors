const pptxgen = require("pptxgenjs");
const fs = require('fs');

const RED = "C8102E";
const DARKRED = "9E0E24";
const ORANGE = "CC5A1E";
const SIDEBAR_BG = "F2F2F2";
const NAV_INACTIVE = "EBB7BE";
const NAV_INACTIVE_TEXT = "D98A94";
const BODY = "3A3A3A";
const SUBHEAD = "555555";
const PLACEHOLDER_BG = "E9E9E9";
const PLACEHOLDER_BORDER = "AFAFAF";
const WHITE = "FFFFFF";

const PAGE_W = 13.333;
const PAGE_H = 7.5;
const SIDEBAR_W = 1.9;
const HEADER_H = 1.2;
const CONTENT_X = SIDEBAR_W + 0.3;
const CONTENT_W = PAGE_W - CONTENT_X - 0.4;
const CONTENT_Y0 = HEADER_H + 0.18;

const NAV_ITEMS = [
  "Project Overview",
  "Data Sources",
  "Exploratory Data\nAnalysis",
  "Data Preprocessing",
  "AQFM Model &\nBaseline",
  "Evaluation &\nSummary",
];

const HEAD_FONT = "Cambria";
const BODY_FONT = "Calibri";

let pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: PAGE_W, height: PAGE_H });
pres.layout = "WIDE";

function addLogo(slide) {
  slide.addShape("ellipse", {
    x: 0.28, y: 0.18, w: 0.8, h: 0.8,
    fill: { color: RED }, line: { color: WHITE, width: 2 },
  });
  slide.addShape("ellipse", {
    x: 0.45, y: 0.35, w: 0.46, h: 0.46,
    fill: { color: ORANGE }, line: { type: "none" },
  });
  slide.addText("AQ\nFM", {
    x: 0.28, y: 0.18, w: 0.8, h: 0.8, isTextBox: true, margin: 0,
    align: "center", valign: "middle", fontFace: HEAD_FONT,
    fontSize: 10, bold: true, color: WHITE, lineSpacing: 11,
  });
}

function addChrome(slide, activeIdx) {
  slide.background = { color: WHITE };
  // top header strip (full width)
  slide.addShape("rect", {
    x: 0, y: 0, w: PAGE_W, h: HEADER_H, fill: { color: SIDEBAR_BG }, line: { type: "none" },
  });
  // sidebar (below header, full height)
  slide.addShape("rect", {
    x: 0, y: HEADER_H, w: SIDEBAR_W, h: PAGE_H - HEADER_H, fill: { color: SIDEBAR_BG }, line: { type: "none" },
  });
  addLogo(slide);
  if (activeIdx >= 0) {
    let y = 1.55;
    NAV_ITEMS.forEach((item, i) => {
      const isActive = i === activeIdx;
      slide.addText(item, {
        x: 0.22, y: y, w: SIDEBAR_W - 0.4, h: 0.55, isTextBox: true, margin: 0,
        align: "left", valign: "top", fontFace: BODY_FONT,
        fontSize: 10.5, bold: isActive, color: isActive ? RED : NAV_INACTIVE_TEXT,
        lineSpacing: 12,
      });
      y += 0.82;
    });
  }
}

function addHeaderTitle(slide, title, opts) {
  opts = opts || {};
  slide.addText(title, {
    x: CONTENT_X, y: 0.12, w: CONTENT_W, h: HEADER_H - 0.24, isTextBox: true, margin: 0,
    align: "left", valign: "middle", fontFace: HEAD_FONT,
    fontSize: opts.fontSize || 24, bold: true, color: RED,
  });
}

function addBar(slide, x, y, w, h, text, color, fontSize) {
  slide.addShape("rect", { x, y, w, h, fill: { color: color }, line: { type: "none" } });
  slide.addText(text, {
    x: x + 0.08, y: y, w: w - 0.16, h: h, isTextBox: true, margin: 0,
    align: "left", valign: "middle", fontFace: BODY_FONT,
    fontSize: fontSize || 12, bold: true, color: WHITE,
  });
}

const PLOTS = "C:/Users/phala/.gemini/antigravity-ide/brain/22bc3195-0369-4654-91bf-d188974f2998/scratch/plots/";
const ARTIF = "C:/Users/phala/.gemini/antigravity-ide/brain/22bc3195-0369-4654-91bf-d188974f2998/";
const PROJ = "C:/Users/phala/Foundation Model/";

const IMG_MAP = {
    "station_map.png": PLOTS + "station_map.png",
    "pm25_timeseries.png": PLOTS + "pm25_timeseries.png",
    "diurnal_pattern.png": PLOTS + "diurnal_pattern.png",
    "pm25_distribution.png": PLOTS + "pm25_distribution.png",
    "correlation_heatmap.png": PLOTS + "correlation_heatmap.png",
    "missingness_heatmap.png": PLOTS + "missingness_heatmap.png",
    "preprocessing_pipeline.jpg": ARTIF + "preprocessing_pipeline_1789624984568.jpg",
    "ttm_forecast_vs_truth.png": PROJ + "_archive_old/Zeroshot forecast/ttm_forecast_vs_truth.png",
    "aqfm_architecture.jpg": ARTIF + "aqfm_architecture_1789624971981.jpg",
};

function addPlaceholder(slide, x, y, w, h, label) {
  const filename = label.split("\\n")[0].trim();
  const realPath = IMG_MAP[filename];
  if (realPath && fs.existsSync(realPath)) {
      slide.addImage({ path: realPath, x: x, y: y, w: w, h: h });
  } else {
      slide.addShape("rect", {
        x, y, w, h, fill: { color: PLACEHOLDER_BG },
        line: { color: PLACEHOLDER_BORDER, width: 1, dashType: "dash" },
      });
      slide.addText([
        { text: "IMAGE PLACEHOLDER\n", options: { bold: true, fontSize: 11, color: "777777" } },
        { text: label, options: { italic: true, fontSize: 9.5, color: "999999" } },
      ], {
        x: x + 0.2, y: y, w: w - 0.4, h: h, isTextBox: true, margin: 0,
        align: "center", valign: "middle", fontFace: BODY_FONT, lineSpacing: 14,
      });
  }
}

function bodyPara(text, opts) {
  opts = opts || {};
  return {
    x: opts.x, y: opts.y, w: opts.w, h: opts.h, isTextBox: true, margin: 2,
    align: "left", valign: "top", fontFace: BODY_FONT,
    fontSize: opts.fontSize || 11, color: opts.color || BODY, lineSpacing: opts.lineSpacing || 14,
  };
}

// ---------------------------------------------------------------
// SLIDE 1: Title
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addShape("rect", { x: 0, y: 0, w: PAGE_W, h: HEADER_H, fill: { color: SIDEBAR_BG }, line: { type: "none" } });
  slide.addShape("rect", { x: 0, y: HEADER_H, w: SIDEBAR_W, h: PAGE_H - HEADER_H, fill: { color: SIDEBAR_BG }, line: { type: "none" } });
  addLogo(slide);
  let y = 1.55;
  NAV_ITEMS.forEach((item) => {
    slide.addText(item, {
      x: 0.22, y: y, w: SIDEBAR_W - 0.4, h: 0.55, isTextBox: true, margin: 0,
      align: "left", valign: "top", fontFace: BODY_FONT,
      fontSize: 10.5, bold: false, color: NAV_INACTIVE_TEXT, lineSpacing: 12,
    });
    y += 0.82;
  });

  slide.addText("Air Quality Forecasting\nusing Foundation Models", {
    x: CONTENT_X, y: 2.3, w: CONTENT_W, h: 1.6, isTextBox: true, margin: 0,
    align: "center", valign: "middle", fontFace: HEAD_FONT,
    fontSize: 34, bold: true, color: RED, lineSpacing: 40,
  });
  slide.addText("A Spatio-Temporal Deep Learning Approach for Multi-Pollutant Forecasting in Delhi", {
    x: CONTENT_X, y: 3.85, w: CONTENT_W, h: 0.6, isTextBox: true, margin: 0,
    align: "center", valign: "middle", fontFace: HEAD_FONT,
    fontSize: 17, bold: true, color: RED,
  });
  slide.addText("Delhi CPCB Study  |  Progress Report  |  September 2026", {
    x: CONTENT_X, y: 5.2, w: CONTENT_W, h: 0.5, isTextBox: true, margin: 0,
    align: "center", valign: "middle", fontFace: BODY_FONT,
    fontSize: 14, color: SUBHEAD,
  });
}

// ---------------------------------------------------------------
// SLIDE 2: Project Overview
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 0);
  addHeaderTitle(slide, "Project Overview");

  slide.addText("A Spatio-Temporal Foundation Model for Multi-Pollutant Forecasting", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 13, bold: true, italic: true, color: ORANGE,
  });

  slide.addShape("rect", { x: CONTENT_X, y: CONTENT_Y0 + 0.42, w: CONTENT_W, h: 1.15, fill: { color: SIDEBAR_BG }, line: { type: "none" } });
  slide.addText([
    { text: "The Problem: ", options: { bold: true, color: RED } },
    { text: "Delhi experiences severe, recurring pollution episodes (PM2.5 > 250 \u00b5g/m\u00b3). Authorities need reliable 1 to 3-day forecasts to trigger early interventions (e.g., GRAP).\n\n", options: {} },
    { text: "The Research Gap: ", options: { bold: true, color: RED } },
    { text: "Existing air quality models typically forecast a single pollutant at a single station. They fail to leverage the physical reality that pollutants travel with the wind (spatial relationships) and react with each other (chemical relationships). When sensors go offline, traditional models fail or rely on flawed pre-imputation.", options: {} },
  ], {
    x: CONTENT_X + 0.15, y: CONTENT_Y0 + 0.42, w: CONTENT_W - 0.3, h: 1.15, isTextBox: true, margin: 4,
    fontFace: BODY_FONT, fontSize: 10.5, color: BODY, valign: "top", lineSpacing: 13,
  });

  const colY = CONTENT_Y0 + 1.72;
  const colH = PAGE_H - colY - 0.3;
  const colW = (CONTENT_W - 0.4) / 3;
  const cols = [
    {
      color: ORANGE, title: "WHAT are we building?",
      body: "A deep learning foundation model (AQFM) that ingests irregularly sampled data from 37 stations and predicts multiple pollutants simultaneously up to 72 hours in advance.\n\nIt produces calibrated probabilistic forecasts, telling us not just what pollution will be, but how certain it is.",
    },
    {
      color: RED, title: "WHY this approach?",
      body: "Foundation models offer massive scalability; a model trained on Delhi can be scaled to all over India without retraining from scratch.\n\nFurthermore, unlike standard deep learning models, it can natively handle real-time streaming data that has missing values and missing variables, ensuring it doesn't break when sensors go offline.",
    },
    {
      color: ORANGE, title: "HOW are we doing it?",
      body: "1. Collect 8 years of CPCB station data & ERA5 weather.\n2. Clean and unify into model-ready tensors.\n3. Build an Axial Transformer with a Query-Based Decoder.\n4. Train with realistic 'mixed masking' to simulate outages.\n5. Evaluate against operational baselines like XGBoost.",
    },
  ];
  cols.forEach((c, i) => {
    const x = CONTENT_X + i * (colW + 0.2);
    addBar(slide, x, colY, colW, 0.4, c.title, c.color, 10.5);
    slide.addText(c.body, {
      x: x, y: colY + 0.48, w: colW, h: colH - 0.48, isTextBox: true, margin: 3,
      fontFace: BODY_FONT, fontSize: 9.5, color: BODY, valign: "top", lineSpacing: 12,
    });
  });
}

// ---------------------------------------------------------------
// SLIDE 3: Our Data Sources
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 1);
  addHeaderTitle(slide, "Our Data Sources");

  slide.addText("We use two complementary datasets: ground-level measurements (CPCB) and satellite weather reanalysis (ERA5)", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });

  const colW = (CONTENT_W - 0.3) / 2;
  const colY = CONTENT_Y0 + 0.45;
  // Source 1
  addBar(slide, CONTENT_X, colY, colW, 0.4, "Source 1: CPCB Ground Stations", ORANGE, 11);
  slide.addText([
    { text: "CPCB (Central Pollution Control Board) operates a network of automatic air quality monitoring stations across Delhi NCR. Each station measures pollutant concentrations and local weather every 15 minutes, 24/7.\n\n", options: {} },
    { text: "Pollutants measured (9): ", options: { bold: true } },
    { text: "PM2.5, PM10, NO, NO2, NOx, Ozone, SO2, NH3, CO\n", options: {} },
    { text: "Weather measured (7): ", options: { bold: true } },
    { text: "Temperature, Humidity, Wind Speed, Wind Direction, Solar Radiation, Pressure, Rainfall\n", options: {} },
    { text: "Also measured: ", options: { bold: true } },
    { text: "Benzene\n\n", options: {} },
    { text: "PROBLEM: ", options: { bold: true, color: RED } },
    { text: "Sensors frequently go offline. Some stations have 15-35% missing data. Entire stations sometimes go dark for days or weeks. This is not random -- when one sensor fails, often ALL sensors at that station fail together (systemic block missingness).", options: {} },
  ], {
    x: CONTENT_X, y: colY + 0.48, w: colW, h: 2.55, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.7, color: BODY, valign: "top", lineSpacing: 12.5,
  });

  // stats row under source 1
  const stats1 = [
    ["37", "Active Stations"], ["2018-2025", "Years Covered"], ["~155M", "Total Observations"],
  ];
  const statY = colY + 3.15;
  const statW = colW / 3;
  stats1.forEach((s, i) => {
    slide.addText([
      { text: s[0] + "\n", options: { bold: true, color: RED, fontSize: 17 } },
      { text: s[1], options: { fontSize: 8.5, color: SUBHEAD } },
    ], {
      x: CONTENT_X + i * statW, y: statY, w: statW, h: 0.75, isTextBox: true, margin: 2,
      align: "center", valign: "top", fontFace: BODY_FONT, lineSpacing: 13,
    });
  });

  // Source 2
  const x2 = CONTENT_X + colW + 0.3;
  addBar(slide, x2, colY, colW, 0.4, "Source 2: ERA5 Weather Reanalysis", RED, 11);
  slide.addText([
    { text: "ERA5 is produced by ECMWF (European weather agency). It combines satellite observations, weather station data, and physics-based models to produce the best estimate of 'what the weather actually was' at every point on Earth, every hour, going back to 1940.\n\n", options: {} },
    { text: "We fetched ERA5 data for the Delhi region (2017-2025) and interpolated it to each of our 37 station locations. Key variables:\n", options: {} },
    { text: "Temperature, Humidity, Wind (U+V), Pressure, Solar Radiation, Rainfall\n\n", options: { bold: true } },
    { text: "Boundary Layer Height (BLH) -- THE critical variable\n", options: { bold: true, color: RED } },
    { text: "BLH tells us how high pollution can mix in the atmosphere. On winter mornings when BLH is low (~100m), pollution gets trapped at ground level. CPCB stations do NOT measure BLH. Only ERA5 provides it.", options: {} },
  ], {
    x: x2, y: colY + 0.48, w: colW, h: 2.55, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.7, color: BODY, valign: "top", lineSpacing: 12.5,
  });

  const stats2 = [["15 min", "Temporal Resolution"], ["0%", "Missing Data"], ["37 sites", "Spatial Coverage"]];
  stats2.forEach((s, i) => {
    slide.addText([
      { text: s[0] + "\n", options: { bold: true, color: RED, fontSize: 17 } },
      { text: s[1], options: { fontSize: 8.5, color: SUBHEAD } },
    ], {
      x: x2 + i * statW, y: statY, w: statW, h: 0.75, isTextBox: true, margin: 2,
      align: "center", valign: "top", fontFace: BODY_FONT, lineSpacing: 13,
    });
  });
}

// ---------------------------------------------------------------
// SLIDE 4: 37 Monitoring Stations
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 1);
  addHeaderTitle(slide, "37 Monitoring Stations Across Delhi NCR");

  slide.addText("Each dot is a station. Color shows how polluted it is on average (Mean PM2.5 concentration)", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });

  const imgW = 6.3;
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, imgW, 4.65, "station_map.png\n(map of 37 CPCB stations, colored by mean PM2.5)");

  const tblX = CONTENT_X + imgW + 0.3;
  const tblW = CONTENT_W - imgW - 0.3;
  addBar(slide, tblX, CONTENT_Y0 + 0.45, tblW, 0.38, "Dataset at a Glance", RED, 11);

  const rows = [
    ["Stations", "37 (operated by DPCC, IITM, CPCB)"],
    ["Time Span", "Jan 2018 -- Dec 2025 (8 years)"],
    ["Sample Rate", "Every 15 minutes (96 readings/day)"],
    ["Timesteps", "280,512 per station"],
    ["Pollutants", "PM2.5, PM10, NO, NO2, NOx, Ozone, SO2, NH3, CO"],
    ["Met Variables", "Temp, Humidity, Wind, Pressure, Rain, Solar Rad"],
    ["Total Channels", "17 measured (+2 derived = 19 for model)"],
    ["Total Cells", "176,442,048 (stations x channels x time)"],
    ["Valid Obs", "154,926,304 (87.8% observed)"],
    ["Missing", "21,515,744 (12.2% gaps)"],
  ];
  const tableRows = rows.map(([k, v]) => [
    { text: k, options: { bold: true, color: RED, fontSize: 9, fill: { color: SIDEBAR_BG } } },
    { text: v, options: { fontSize: 9, color: BODY } },
  ]);
  slide.addTable(tableRows, {
    x: tblX, y: CONTENT_Y0 + 0.9, w: tblW, colW: [1.35, tblW - 1.35],
    fontFace: BODY_FONT, autoPage: false, border: { type: "solid", color: "DDDDDD", pt: 0.5 },
    valign: "middle", margin: 2,
  });
}

// ---------------------------------------------------------------
// SLIDE 5: Section divider - EDA
// ---------------------------------------------------------------
function dividerSlide(activeIdx, title, subtitleLines) {
  let slide = pres.addSlide();
  addChrome(slide, activeIdx);
  slide.addText(title, {
    x: CONTENT_X, y: 2.9, w: CONTENT_W, h: 1.0, isTextBox: true, margin: 0,
    align: "left", valign: "middle", fontFace: HEAD_FONT,
    fontSize: 34, bold: true, color: RED,
  });
  let y = 3.95;
  subtitleLines.forEach((line) => {
    slide.addText(line.text, {
      x: CONTENT_X, y: y, w: CONTENT_W, h: 0.45, isTextBox: true, margin: 0,
      align: "left", valign: "middle", fontFace: BODY_FONT,
      fontSize: line.size || 15, italic: line.italic || false, bold: line.bold || false, color: line.color || SUBHEAD,
    });
    y += 0.5;
  });
  slide.addShape("line", { x: CONTENT_X, y: 2.75, w: 1.2, h: 0, line: { color: ORANGE, width: 3 } });
  return slide;
}

dividerSlide(2, "Exploratory Data Analysis", [
  { text: "Understanding our data before we build anything" },
]);

// ---------------------------------------------------------------
// SLIDE 6: EDA PM2.5 over 8 years
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 2);
  addHeaderTitle(slide, "EDA: PM2.5 Concentration Over 8 Years");
  slide.addText("Each graph shows daily PM2.5 levels at one station. Notice the dramatic spikes every winter (Oct-Jan).", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  const imgW = 6.9;
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, imgW, 4.65, "pm25_timeseries.png\n(daily PM2.5 line charts per station, 2018-2025)");

  const rx = CONTENT_X + imgW + 0.3;
  const rw = CONTENT_W - imgW - 0.3;
  addBar(slide, rx, CONTENT_Y0 + 0.45, rw, 0.38, "What This Tells Us", RED, 11);
  slide.addText([
    { text: "Strong seasonal pattern: ", options: { bold: true } },
    { text: "Every winter (Nov-Jan), PM2.5 rockets above 250 ug/m3.\n\n", options: {} },
    { text: "Red line = 'Severe' AQI threshold (250 ug/m3). These are the dangerous episodes we need to predict.\n\n", options: {} },
    { text: "Summers are relatively clean (30-80 ug/m3). But winters routinely hit 500-700+ ug/m3.\n\n", options: {} },
    { text: "The pattern is consistent across all stations -- this means spatial modelling can help.\n\n", options: {} },
    { text: "Key stats from data:\n", options: { bold: true, color: RED } },
    { text: "Mean PM2.5: 102.9 ug/m3\nMedian: 69.9 ug/m3\n8.07% of readings > 250 (Severe)\n0.67% of readings > 500 (Hazardous)", options: {} },
  ], {
    x: rx, y: CONTENT_Y0 + 0.9, w: rw, h: 4.2, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.7, color: BODY, valign: "top", lineSpacing: 12.5,
  });
}

// ---------------------------------------------------------------
// SLIDE 7: EDA Daily Rhythm
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 2);
  addHeaderTitle(slide, "EDA: The Daily Rhythm of Pollution");
  slide.addText("Average pollutant concentrations by hour of day, computed from all 37 stations over 8 years of real data", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  const imgW = 6.9;
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, imgW, 3.6, "diurnal_pattern.png\n(hourly average concentration curves)");

  const rx = CONTENT_X + imgW + 0.3;
  const rw = CONTENT_W - imgW - 0.3;
  addBar(slide, rx, CONTENT_Y0 + 0.45, rw, 0.38, "What This Tells Us", RED, 11);
  slide.addText([
    { text: "PM2.5 (teal line) is highest at night and early morning, and lowest around 4-5 PM.\n\n", options: {} },
    { text: "Ozone (orange line) shows the OPPOSITE pattern -- highest at midday. This is because Ozone is produced by sunlight reacting with NOx.\n\n", options: {} },
    { text: "This is important because it means PM2.5 and Ozone are driven by different physics. A good model must understand both patterns.\n\n", options: {} },
    { text: "The daily cycle is so strong that a simple 'same hour yesterday' baseline is surprisingly effective.", options: {} },
  ], {
    x: rx, y: CONTENT_Y0 + 0.9, w: rw, h: 3.1, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.7, color: BODY, valign: "top", lineSpacing: 12.5,
  });

  addBar(slide, CONTENT_X, CONTENT_Y0 + 4.25, CONTENT_W, 0.75, "", ORANGE, 10);
  slide.addText([
    { text: "WHY THIS MATTERS FOR OUR MODEL:  ", options: { bold: true } },
    { text: "The model needs to learn the diurnal cycle (captured by Fourier time features in the tokeniser), AND the chemistry (captured by cross-channel attention in the encoder, so PM2.5 and Ozone can 'talk to each other' inside the model).", options: {} },
  ], {
    x: CONTENT_X + 0.1, y: CONTENT_Y0 + 4.25, w: CONTENT_W - 0.2, h: 0.75, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.7, color: WHITE, valign: "middle", lineSpacing: 11.5,
  });
}

// ---------------------------------------------------------------
// SLIDE 8: EDA log transform
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 2);
  addHeaderTitle(slide, "EDA: Why We Apply log(1+x) Transform");
  slide.addText("Pollutant concentrations are heavily right-skewed. Applying log(1+x) makes them near-normal, which deep learning models learn much better.", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, CONTENT_W, 3.0, "pm25_distribution.png\n(raw vs. log(1+x)-transformed PM2.5 histograms)");

  const colW = (CONTENT_W - 0.3) / 2;
  const y2 = CONTENT_Y0 + 3.65;
  addBar(slide, CONTENT_X, y2, colW, 0.38, "The Problem (Left Plot)", RED, 10.5);
  slide.addText("Raw PM2.5 data has a long right tail -- most readings are low (median 69.9 ug/m3) but some reach 500-1000 ug/m3. The mean (102.9) is much higher than the median. Neural networks struggle with this kind of skewed data because the extreme values dominate the loss function and make training unstable.", {
    x: CONTENT_X, y: y2 + 0.46, w: colW, h: 1.2, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.5, color: BODY, valign: "top", lineSpacing: 12,
  });
  const x2c = CONTENT_X + colW + 0.3;
  addBar(slide, x2c, y2, colW, 0.38, "The Solution (Right Plot)", ORANGE, 10.5);
  slide.addText("After applying log(1+x), the distribution is nearly symmetric and bell-shaped. Mean and median are now very close (both ~4.26). This makes the data much easier for deep learning models to learn from. We apply this transform to ALL pollutant channels before training, and invert it after prediction to get real concentrations.", {
    x: x2c, y: y2 + 0.46, w: colW, h: 1.2, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.5, color: BODY, valign: "top", lineSpacing: 12,
  });
}

// ---------------------------------------------------------------
// SLIDE 9: Cross-pollutant correlations
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 2);
  addHeaderTitle(slide, "EDA: Cross-Pollutant Correlations");
  slide.addText("This heatmap shows how each pollutant relates to every other. Computed from 1.84 million simultaneous observations across all stations.", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  const imgW = 5.4;
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, imgW, 4.65, "correlation_heatmap.png\n(pollutant x pollutant correlation matrix)");

  const rx = CONTENT_X + imgW + 0.3;
  const rw = CONTENT_W - imgW - 0.3;
  addBar(slide, rx, CONTENT_Y0 + 0.45, rw, 0.38, "Key Takeaways", RED, 11);
  slide.addText([
    { text: "1. PM2.5 correlates extremely strongly with PM10 (r=0.84)\n", options: { bold: true } },
    { text: "   They share the same underlying physics and emission sources.\n\n", options: {} },
    { text: "2. PM2.5 correlates strongly with CO (r=0.43)\n", options: { bold: true } },
    { text: "   Both are products of combustion (vehicles, biomass burning).\n\n", options: {} },
    { text: "3. Ozone negatively correlates with CO (r=-0.19) and PM10 (r=-0.08)\n", options: { bold: true } },
    { text: "   Ozone is produced by sunlight, while particulates and CO accumulate when the boundary layer is trapped (no wind/sun).\n\n", options: {} },
    { text: "4. AT (Temperature) is negatively correlated with PM10 (r=-0.25)\n", options: { bold: true } },
    { text: "   Cold weather = lower boundary layer = trapped pollution.\n\n", options: {} },
    { text: "WHY THIS MATTERS: ", options: { bold: true, color: RED } },
    { text: "These cross-pollutant relationships mean forecasting PM2.5 gets better if the model also sees CO, NOx, and weather. Our AQFM model learns all channels simultaneously. We also IMPOSE known physical laws (PM2.5 < PM10 always, NO + NO2 = NOx) directly into the model's architecture.", options: {} },
  ], {
    x: rx, y: CONTENT_Y0 + 0.9, w: rw, h: 4.2, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.3, color: BODY, valign: "top", lineSpacing: 11.5,
  });
}

// ---------------------------------------------------------------
// SLIDE 10: Missingness
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 2);
  addHeaderTitle(slide, "EDA: Data Quality \u2014 Missingness by Channel and Year", { fontSize: 21 });
  slide.addText("This heatmap shows what percentage of readings are actually present for each channel, each year. Green = good coverage, yellow = partial.", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  const imgW = 6.3;
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, imgW, 4.65, "missingness_heatmap.png\n(coverage % per channel per year)");

  const rx = CONTENT_X + imgW + 0.3;
  const rw = CONTENT_W - imgW - 0.3;
  addBar(slide, rx, CONTENT_Y0 + 0.45, rw, 0.38, "What This Tells Us", RED, 11);
  slide.addText([
    { text: "Primary pollutants (PM2.5, PM10, NO2, etc.) have 75-93% coverage. That means 7-25% missing per year.\n\n", options: {} },
    { text: "Coverage is IMPROVING over time -- 2024-2025 has 90%+ on most channels.\n\n", options: {} },
    { text: "SO2, NH3, Benzene have worse coverage (~55-73%) because fewer stations measure them.\n\n", options: {} },
    { text: "All weather channels (AT through TOT_RF) show 100% -- this is AFTER we filled their gaps using ERA5.\n\n", options: {} },
    { text: "KEY DESIGN DECISION: ", options: { bold: true, color: RED } },
    { text: "We do NOT impute missing values. Because outages are 'systemic' (entire stations offline for days), imputation would just inject hallucinated data into our training signal. By handling missing data natively via binary masks, the model learns to rely on its neighbors (spatial attention) and weather (channel attention) to survive real-world sensor failures.", options: {} },
  ], {
    x: rx, y: CONTENT_Y0 + 0.9, w: rw, h: 4.2, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.4, color: BODY, valign: "top", lineSpacing: 12,
  });
}

// ---------------------------------------------------------------
// SLIDE 11: Divider - Data Preprocessing
// ---------------------------------------------------------------
dividerSlide(3, "Data Preprocessing", [
  { text: "Unifying CPCB and ERA5 Data for Model Ingestion" },
]);

// ---------------------------------------------------------------
// SLIDE 12: 6-step pipeline (steps 1-3)
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 3);
  addHeaderTitle(slide, "The 6-Step Preprocessing Pipeline");
  slide.addText("We built a single script (preprocess_aqfm.py) that takes raw CSV files and produces model-ready numpy arrays", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, CONTENT_W, 2.05, "preprocessing_pipeline.jpg\n(diagram of the 6-step pipeline)");

  const y3 = CONTENT_Y0 + 2.65;
  const colW = (CONTENT_W - 0.4) / 3;
  const steps = [
    { color: ORANGE, title: "Step 1: Load & Align CSVs", body: "300+ CSV files (one per station per year) are read, aligned to a single 15-minute timeline from Jan 2018 to Dec 2025, and stacked into one consistent grid. Duplicate timestamps are removed, negative readings are treated as missing." },
    { color: RED, title: "Step 2: Fill Weather Gaps with ERA5", body: "CPCB weather sensors (temperature, humidity, wind, etc.) have significant gaps. ERA5 reanalysis fills these gaps perfectly -- all 7 meteorological channels are now at 100% coverage. 22+ million individual gap values were filled." },
    { color: ORANGE, title: "Step 3: Physical Sanity Checks", body: "Clip physically impossible values: e.g., negative concentrations, temperature outside [-15, 55] C, wind speed > 50 m/s. Also flag PM2.5 > PM10 violations (0.44% of data) which the model handles via its physical constraint architecture." },
  ];
  steps.forEach((s, i) => {
    const x = CONTENT_X + i * (colW + 0.2);
    addBar(slide, x, y3, colW, 0.6, s.title, s.color, 9.5);
    slide.addText(s.body, {
      x: x, y: y3 + 0.68, w: colW, h: 1.5, isTextBox: true, margin: 3,
      fontFace: BODY_FONT, fontSize: 9, color: BODY, valign: "top", lineSpacing: 11,
    });
  });
}

// ---------------------------------------------------------------
// SLIDE 13: preprocessing steps 4-6 + output table
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 3);
  addHeaderTitle(slide, "Preprocessing: Steps 4-6 and Final Output", { fontSize: 22 });
  slide.addText("The remaining steps produce the exact arrays the model expects as input", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });

  const y3 = CONTENT_Y0 + 0.4;
  const colW = (CONTENT_W - 0.4) / 3;
  const steps = [
    { color: ORANGE, title: "Step 4: Stack Into Dense Arrays", body: "All data from all 37 stations and 17 channels is stacked into two large tensors: 'values' and 'observed' (a binary mask). Shape: (37 stations, 17 channels, 280512 timesteps)." },
    { color: RED, title: "Step 5: Compute Static Features", body: "For each station, we compute 4 fixed properties: (1) average PM2.5, (2) average PM10, (3) elevation proxy from pressure, (4) station density (stations within 5km)." },
    { color: ORANGE, title: "Step 6: Fit Outage Distribution", body: "We fit a LogNormal distribution to 125,506 real outages (median 0.5 hours), so training simulates realistic missing-data patterns, not artificial ones." },
  ];
  steps.forEach((s, i) => {
    const x = CONTENT_X + i * (colW + 0.2);
    addBar(slide, x, y3, colW, 0.55, s.title, s.color, 9.3);
    slide.addText(s.body, {
      x: x, y: y3 + 0.62, w: colW, h: 1.05, isTextBox: true, margin: 3,
      fontFace: BODY_FONT, fontSize: 8.7, color: BODY, valign: "top", lineSpacing: 10.5,
    });
  });

  const tblY = y3 + 1.85;
  addBar(slide, CONTENT_X, tblY, CONTENT_W, 0.36, "Final Output Files  (saved in preprocessed/ folder -- these are what the model will consume)", RED, 10);

  const header = ["File", "Shape", "Size", "Description"].map((t) => ({
    text: t, options: { bold: true, color: WHITE, fill: { color: ORANGE }, fontSize: 9.5 },
  }));
  const dataRows = [
    ["values.npy", "(37 x 17 x 280,512)", "706 MB", "All pollutant and weather readings. Missing values = 0."],
    ["observed.npy", "(37 x 17 x 280,512)", "706 MB", "Binary mask: 1 = real measurement, 0 = missing."],
    ["timestamps_h.npy", "(280,512)", "2.2 MB", "Hours since Jan 1, 2018. Model uses this for time encoding."],
    ["coords.npy", "(37 x 2)", "< 1 KB", "Latitude and longitude of each station."],
    ["static.npy", "(37 x 4)", "< 1 KB", "Static features per station (mean PM, density, etc.)"],
  ].map((row) => row.map((t) => ({ text: t, options: { fontSize: 8.8, color: BODY } })));
  slide.addTable([header, ...dataRows], {
    x: CONTENT_X, y: tblY + 0.4, w: CONTENT_W, colW: [1.6, 1.9, 0.9, CONTENT_W - 4.4],
    fontFace: BODY_FONT, autoPage: false, border: { type: "solid", color: "DDDDDD", pt: 0.5 },
    valign: "middle", margin: 2, rowH: 0.34,
  });
}

// ---------------------------------------------------------------
// SLIDE 14: Baseline TTM
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 4);
  addHeaderTitle(slide, "Baseline: Zero-Shot Forecasting with IBM Granite TTM", { fontSize: 19 });
  slide.addText("Before building our own model, we tested an existing pre-trained time-series foundation model on our data to establish a baseline performance.", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  const imgW = 6.7;
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.45, imgW, 4.65, "ttm_forecast_vs_truth.png\n(zero-shot forecast vs. ground truth)");

  const rx = CONTENT_X + imgW + 0.3;
  const rw = CONTENT_W - imgW - 0.3;
  addBar(slide, rx, CONTENT_Y0 + 0.45, rw, 0.38, "What We Did", RED, 11);
  slide.addText("We used IBM's Granite Time-Series Tiny Time Mixer (TTM), a pre-trained foundation model for time-series forecasting. We ran it in ZERO-SHOT mode -- meaning we did NOT train or fine-tune it on our data at all. We just gave it 21 days of history and asked it to forecast the next 4 days.", {
    x: rx, y: CONTENT_Y0 + 0.9, w: rw, h: 1.35, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.5, color: BODY, valign: "top", lineSpacing: 12,
  });

  const statY = CONTENT_Y0 + 2.35;
  const statW = rw / 2;
  const stats = [["32.6 ug/m3", "PM2.5 Mean Absolute Error"], ["61.8 ug/m3", "PM10 Mean Absolute Error"]];
  stats.forEach((s, i) => {
    slide.addShape("rect", { x: rx + i * (statW), y: statY, w: statW - 0.1, h: 0.85, fill: { color: SIDEBAR_BG }, line: { type: "none" } });
    slide.addText([
      { text: s[0] + "\n", options: { bold: true, color: RED, fontSize: 16 } },
      { text: s[1], options: { fontSize: 8.3, color: SUBHEAD } },
    ], {
      x: rx + i * statW, y: statY, w: statW - 0.1, h: 0.85, isTextBox: true, margin: 2,
      align: "center", valign: "middle", fontFace: BODY_FONT, lineSpacing: 13,
    });
  });

  addBar(slide, rx, statY + 0.98, rw, 0.36, "Key Observations", ORANGE, 10.5);
  slide.addText([
    { text: "The model correctly learns the diurnal (daily) cycle even without being trained on this specific dataset.\n\n", options: {} },
    { text: "BUT it fails completely on the magnitude scale. It wildly overpredicts all concentrations, forecasting values way above extreme events for almost all data points.\n\n", options: {} },
    { text: "This proves off-the-shelf time-series models fail zero-shot on air quality. We need AQFM: a custom model designed to understand realistic concentration limits and physical constraints.", options: {} },
  ], {
    x: rx, y: statY + 1.4, w: rw, h: 1.7, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9, color: BODY, valign: "top", lineSpacing: 11.5,
  });
}

// ---------------------------------------------------------------
// SLIDE 15: Divider - AQFM Model
// ---------------------------------------------------------------
dividerSlide(4, "The AQFM Model", [
  { text: "What we are building next", bold: true, color: RED, size: 16 },
  { text: "(Air Quality Foundation Model \u2014 Spatio-Temporal Transformer)", italic: true, size: 13 },
]);

// ---------------------------------------------------------------
// SLIDE 16: AQFM architecture
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 4);
  addHeaderTitle(slide, "AQFM Architecture \u2014 How the Model Works", { fontSize: 22 });
  slide.addText("Data flows from left to right: raw measurements go in, probabilistic forecasts come out", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });
  addPlaceholder(slide, CONTENT_X, CONTENT_Y0 + 0.4, CONTENT_W, 2.0, "aqfm_architecture.jpg\n(end-to-end model architecture diagram)");

  const y3 = CONTENT_Y0 + 2.6;
  const colW = (CONTENT_W - 0.6) / 4;
  const steps = [
    { title: "1. Tokenisation", color: ORANGE, body: "Each 4-hour window of a single pollutant at a single station becomes one 'token'. Missing data is handled by including a mask alongside each token -- the model knows exactly what is real vs missing." },
    { title: "2. Axial Encoder", color: RED, body: "The encoder runs attention in three directions: across TIME, across CHANNELS, and across SPACE. This is much more efficient than attending to everything at once." },
    { title: "3. Query-Based Decoder", color: ORANGE, body: "We give the decoder a 'query': which pollutant, at which station, how far into the future? This means we can forecast any subset of pollutants at any horizon." },
    { title: "4. Probabilistic Output", color: RED, body: "The model outputs a probability distribution (mixture of 3 log-normal distributions) -- not just a value, but a confidence range. Critical for operational decisions." },
  ];
  steps.forEach((s, i) => {
    const x = CONTENT_X + i * (colW + 0.2);
    addBar(slide, x, y3, colW, 0.55, s.title, s.color, 9.3);
    slide.addText(s.body, {
      x: x, y: y3 + 0.62, w: colW, h: 1.5, isTextBox: true, margin: 3,
      fontFace: BODY_FONT, fontSize: 8.6, color: BODY, valign: "top", lineSpacing: 10.5,
    });
  });
}

// ---------------------------------------------------------------
// SLIDE 17: Key design decisions
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 4);
  addHeaderTitle(slide, "Key Design Decisions \u2014 Why AQFM Is Different", { fontSize: 21 });
  slide.addText("Three things that make this model specifically suited for air quality, based on what we learned from the EDA", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: SUBHEAD,
  });

  const y3 = CONTENT_Y0 + 0.42;
  const colW = (CONTENT_W - 0.4) / 3;
  const colH = PAGE_H - y3 - 0.3;
  const decisions = [
    {
      color: ORANGE, title: "Physical Constraints Built Into the Architecture",
      body: "We discovered that PM2.5 is always a subset of PM10 (by physical definition), and NO + NO2 always equals NOx (by chemistry).\n\nInstead of hoping the model learns these rules from data, we ENCODE them directly: the model predicts PM10 and then a fraction (0 to 1) to get PM2.5. This makes PM2.5 > PM10 physically IMPOSSIBLE.\n\nSimilarly, the model predicts NOx and a split-fraction for NO vs NO2. Zero additional parameters. Zero nonsense output.",
    },
    {
      color: RED, title: "Wind-Aware Spatial Attention",
      body: "A station 8km UPWIND is far more informative than one 8km DOWNWIND. Standard models only use distance between stations. Ours adds the bearing relative to the wind direction as an attention bias.\n\nThis means: when the wind blows from the northwest, the model automatically pays more attention to northwest stations. This is a physics-informed inductive bias that distance-only graphs cannot express. The AQFM design spec calls this the 'wind-aware spatial bias MLP' and it operates per attention head.",
    },
    {
      color: ORANGE, title: "Training with Realistic Missing Data Patterns",
      body: "From our EDA, we know that data gaps are SYSTEMIC -- when one sensor fails, all sensors at that station fail together for hours or days.\n\nDuring training, we simulate three types of missingness: (1) mask the future (50% -- this IS forecasting), (2) mask entire stations for realistic durations fitted from real outage data (35%), and (3) drop random channels (15% -- forces the model to forecast PM2.5 from wind and neighbouring stations alone).",
    },
  ];
  decisions.forEach((d, i) => {
    const x = CONTENT_X + i * (colW + 0.2);
    addBar(slide, x, y3, colW, 0.7, d.title, d.color, 9.5);
    slide.addText(d.body, {
      x: x, y: y3 + 0.78, w: colW, h: colH - 0.78, isTextBox: true, margin: 3,
      fontFace: BODY_FONT, fontSize: 8.7, color: BODY, valign: "top", lineSpacing: 11,
    });
  });
}

// ---------------------------------------------------------------
// SLIDE 18: Training Plan & Model Configuration
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 4);
  addHeaderTitle(slide, "Training Plan & Model Configuration", { fontSize: 22 });
  slide.addText("Two-phase training curriculum: first learn general patterns with mixed masking, then fine-tune for specific forecast horizons", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 10.8, italic: true, color: SUBHEAD,
  });

  const colW = (CONTENT_W - 0.3) / 2;
  const y3 = CONTENT_Y0 + 0.42;

  addBar(slide, CONTENT_X, y3, colW, 0.36, "Model Configuration", RED, 10.5);
  const cfgRows = [
    ["Context Window", "5.3 days (512 timesteps at 15-min)"],
    ["Forecast Horizon", "Up to 72 hours (288 timesteps)"],
    ["Patch Size", "4 hours (16 timesteps per token)"],
    ["Neighbourhood", "12 nearest stations per sample"],
    ["Model Dimension", "384"],
    ["Attention Heads", "8"],
    ["Axial Encoder Blocks", "4 (= 12 attention layers)"],
    ["Decoder Layers", "4 (self-attn + cross-attn + FFN)"],
    ["Mixture Components", "3 (log-normal mixture)"],
    ["Optimiser", "AdamW, lr=3e-4, weight decay 0.05"],
    ["Normalisation", "Masked RevIN (context-only stats)"],
    ["Total Parameters", "~23 million"],
  ].map(([k, v]) => [
    { text: k, options: { bold: true, color: RED, fontSize: 8.3, fill: { color: SIDEBAR_BG } } },
    { text: v, options: { fontSize: 8.3, color: BODY } },
  ]);
  slide.addTable(cfgRows, {
    x: CONTENT_X, y: y3 + 0.42, w: colW, colW: [1.7, colW - 1.7],
    fontFace: BODY_FONT, autoPage: false, border: { type: "solid", color: "DDDDDD", pt: 0.5 },
    valign: "middle", margin: 1.5, rowH: 0.27,
  });

  const x2 = CONTENT_X + colW + 0.3;
  addBar(slide, x2, y3, colW, 0.36, "Phase 1: Mixed-Masking Pre-Training", ORANGE, 10);
  slide.addText([
    { text: "The model trains on a mixture of three masking strategies:\n", options: {} },
    { text: "Tail masking (50%): ", options: { bold: true } },
    { text: "hide the last K patches -- this is pure forecasting.\n", options: {} },
    { text: "Block masking (35%): ", options: { bold: true } },
    { text: "simulate station-wide outages (duration from fitted LogNormal).\n", options: {} },
    { text: "Channel masking (15%): ", options: { bold: true } },
    { text: "drop random channels -- forces multi-modal learning.\n\n", options: {} },
    { text: "This teaches the model both forecasting AND imputation simultaneously. The masking weights are designed so the model sees forecasting tasks most often, since that is our primary goal.", options: {} },
  ], {
    x: x2, y: y3 + 0.42, w: colW, h: 1.85, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 8.6, color: BODY, valign: "top", lineSpacing: 10.5,
  });

  addBar(slide, x2, y3 + 2.4, colW, 0.36, "Phase 2: Horizon Fine-Tuning", RED, 10);
  slide.addText([
    { text: "After the model has learned general patterns, we fine-tune specifically for the forecast horizons we care about operationally:\n", options: {} },
    { text: "1 hour", options: { bold: true } }, { text: " -- real-time monitoring and alerts\n", options: {} },
    { text: "6 hours", options: { bold: true } }, { text: " -- same-day planning\n", options: {} },
    { text: "24 hours", options: { bold: true } }, { text: " -- next-day school closure decisions\n", options: {} },
    { text: "72 hours", options: { bold: true } }, { text: " -- GRAP stage activation planning\n\n", options: {} },
    { text: "Only tail masking (pure causal forecasting). Lower learning rate (5e-5). Strict causal split: no future information leaks into the context window.", options: {} },
  ], {
    x: x2, y: y3 + 2.82, w: colW, h: 1.75, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 8.6, color: BODY, valign: "top", lineSpacing: 10.5,
  });
}

// ---------------------------------------------------------------
// SLIDE 19: Evaluation
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 5);
  addHeaderTitle(slide, "How We Will Evaluate the Model", { fontSize: 23 });
  slide.addText("Standard metrics like MAE are misleading for air quality. We use metrics that matter for real-world operational decisions.", {
    x: CONTENT_X, y: CONTENT_Y0, w: CONTENT_W, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: SUBHEAD,
  });

  slide.addShape("rect", { x: CONTENT_X, y: CONTENT_Y0 + 0.4, w: CONTENT_W, h: 0.62, fill: { color: SIDEBAR_BG }, line: { type: "none" } });
  slide.addText("A pooled yearly MAE looks good because clean summer months (PM2.5 = 30-50) dominate the average and hide the model's failures during the dangerous winter episodes (PM2.5 = 300-600) that the forecast ACTUALLY EXISTS to predict. We therefore stratify ALL metrics by AQI band and season.", {
    x: CONTENT_X + 0.12, y: CONTENT_Y0 + 0.4, w: CONTENT_W - 0.24, h: 0.62, isTextBox: true, margin: 2,
    fontFace: BODY_FONT, fontSize: 9.3, color: BODY, valign: "middle", lineSpacing: 11,
  });

  const colW = (CONTENT_W - 0.3) / 2;
  const y3 = CONTENT_Y0 + 1.2;
  addBar(slide, CONTENT_X, y3, colW, 0.36, "Metrics We Will Report", RED, 10.5);
  slide.addText([
    { text: "\u2022 Skill stratified by AQI band (Good / Moderate / Poor / Severe / Hazardous)\n\n", options: {} },
    { text: "\u2022 Episode detection: can the model predict PM2.5 > 250 ug/m3 at 24h and 48h lead?\n\n", options: {} },
    { text: "\u2022 Calibration: are the model's 90% and 95% prediction intervals actually correct?\n\n", options: {} },
    { text: "\u2022 CRPS by lead time: where does skill decay to climatology (random guessing)?\n\n", options: {} },
    { text: "\u2022 Imputation scored separately from forecasting (different tasks, different failures)", options: {} },
  ], {
    x: CONTENT_X, y: y3 + 0.44, w: colW, h: 3.6, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 9.5, color: BODY, valign: "top", lineSpacing: 12.5,
  });

  const x2 = CONTENT_X + colW + 0.3;
  addBar(slide, x2, y3, colW, 0.36, "Baselines We Must Beat", ORANGE, 10.5);
  const baselines = [
    ["Persistence", "Predict tomorrow = today. Floor; anything below this is broken."],
    ["Seasonal Naive", "Predict = same hour yesterday. Captures diurnal cycle for free."],
    ["Climatology", "Average value for this station, hour, and month. Long-horizon competitor."],
    ["Gradient-Boosted Trees", "XGBoost on lagged features. Known to be VERY strong on this task."],
    ["IBM TTM (our baseline)", "Zero-shot result we already have. PM2.5 MAE = 32.6 at 96h."],
    ["PatchTST / iTransformer", "State-of-art time-series models. Isolates value of spatial attention."],
  ];
  let by = y3 + 0.44;
  baselines.forEach(([name, desc]) => {
    slide.addText([
      { text: name + ": ", options: { bold: true, color: RED } },
      { text: desc, options: { color: BODY } },
    ], {
      x: x2, y: by, w: colW, h: 0.52, isTextBox: true, margin: 2,
      fontFace: BODY_FONT, fontSize: 8.8, valign: "top", lineSpacing: 10.5,
    });
    by += 0.56;
  });
}

// ---------------------------------------------------------------
// SLIDE 20: Summary
// ---------------------------------------------------------------
{
  let slide = pres.addSlide();
  addChrome(slide, 5);
  addHeaderTitle(slide, "Summary & Current Status");

  const colW = (CONTENT_W - 0.3) / 2;
  const y3 = CONTENT_Y0 + 0.1;
  addBar(slide, CONTENT_X, y3, colW, 0.4, "Work Completed", RED, 12);
  const done = [
    "Fetched all CPCB data (37 stations, 2009-2025)",
    "Fetched ERA5 reanalysis (2017-2025, interpolated to stations)",
    "Comprehensive EDA: seasonal patterns, correlations, missingness",
    "Zero-shot baseline with IBM TTM (PM2.5 MAE=32.6 at 96h)",
    "Unified CPCB+ERA5 (all met channels now at 100% coverage)",
    "Full preprocessing pipeline: 6-step script producing model-ready .npy files",
    "All files and work uploaded to shared Drive",
  ];
  slide.addText(done.map((t) => ({ text: "\u2713  " + t, options: { breakLine: true, bullet: false } })), {
    x: CONTENT_X, y: y3 + 0.5, w: colW, h: 4.4, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 10.3, color: BODY, valign: "top", lineSpacing: 22,
  });

  const x2 = CONTENT_X + colW + 0.3;
  addBar(slide, x2, y3, colW, 0.4, "What's Next", ORANGE, 12);
  const next = [
    "Build the AQFM tokeniser (patch tokens + time encoding)",
    "Build the axial encoder (time + channel + space attention)",
    "Build the query-based decoder (Design B)",
    "Implement loss functions (NLL + CRPS with order-statistic trick)",
    "Implement physical constraint module (PM2.5<PM10, NO+NO2=NOx)",
    "Two-phase training on preprocessed data",
    "Full evaluation against all baselines",
  ];
  slide.addText(next.map((t) => ({ text: "\u25b8  " + t, options: { breakLine: true, bullet: false } })), {
    x: x2, y: y3 + 0.5, w: colW, h: 4.4, isTextBox: true, margin: 3,
    fontFace: BODY_FONT, fontSize: 10.3, color: BODY, valign: "top", lineSpacing: 22,
  });

  slide.addText("Thank You", {
    x: CONTENT_X, y: 6.55, w: CONTENT_W, h: 0.6, isTextBox: true, margin: 0,
    align: "center", valign: "middle", fontFace: HEAD_FONT, fontSize: 22, bold: true, color: RED,
  });
}

pres.writeFile({ fileName: "C:/Users/phala/Foundation Model/presentation/AQFM_Presentation_Restyled_v4.pptx" }).then(() => {
  console.log("done");
});
