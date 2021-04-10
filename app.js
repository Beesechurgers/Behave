/*
 * Copyright (c) 2021, Beesechurgers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 */

const cv = require('opencv4nodejs');
const tf = require('@tensorflow/tfjs-node');
const jimp = require('jimp');
const path = require('path');
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);

var START_DETECTION = false;
var camera = null;

const cascade = new cv.CascadeClassifier('haarcascade_frontalface_default.xml');
let model = null;
tf.loadLayersModel(`file://${__dirname}/model_tf/model.json`).then(m => {
    console.log("Model was loaded");
    model = m;
});
const FPS = 24;

app.use(express.static('.'));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '\\index.html'))
});

io.on('connection', (socket) => {
    console.log("Connected");
    socket.on('rec', (start) => {
        START_DETECTION = start;
        if (start) {
            startCamera();
        } else {
            stopCamera();
        }
    })
});

var emotion = "NA";
var gray = null, faces = null, values = new Float32Array(1), area = null, bitImg = null,
    pred = null;
let tfImage = null;
const emots = ["angry", 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
setInterval(() => {
    if (!START_DETECTION) {
        return
    }
    var frame = camera.read().flip(1);
    gray = frame.cvtColor(cv.COLOR_BGR2GRAY);
    faces = cascade.detectMultiScale(gray, 1.3, 5);
    for (let i = 0; i < faces.numDetections.length; i++) {
        const ex = faces.objects[i];
        frame.drawRectangle(new cv.Point2(ex.x, ex.y), new cv.Point2(ex.x + ex.width, ex.y + ex.height), new cv.Vec3(0, 255, 0), 2);

        if (model != null) {
            (async () => {
                area = frame.resize(48, 48);
                values = new Float32Array(48 * 48);
                bitImg = await jimp.create(cv.imencode('.jpg', area));

                let i = 0;
                bitImg.scan(0, 0, bitImg.bitmap.width, bitImg.bitmap.height, (x, y, idx) => {
                    const pixel = jimp.intToRGBA(bitImg.getPixelColor(x, y));
                    pixel.r = pixel.r / 127.0 - 1;
                    pixel.g = pixel.g / 127.0 - 1;
                    pixel.b = pixel.b / 127.0 - 1;
                    pixel.a = pixel.a / 127.0 - 1;
                    values[i + 0] = pixel.r;
                    values[i + 1] = pixel.g;
                    values[i + 2] = pixel.b;
                    i++;
                });

                const outShape = [48, 48, 1];
                tfImage = tf.tensor3d(values, outShape, 'float32');
                tfImage = tfImage.expandDims(0);

                pred = model.predict(tfImage).dataSync();
                emotion = emots[getMaxIndex(pred)];
            })();
        }
        frame.putText(emotion, new cv.Point2(ex.x, ex.y), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Vec3(0, 0, 255), 2);
    }
    const img = cv.imencode('.jpg', frame).toString('base64');
    io.emit('image', img);
}, 1000 / FPS);

server.listen(5000, () => {
    console.log('Listening on 5000');
});

function startCamera() {
    if (camera == null) {
        camera = new cv.VideoCapture(0);
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280);
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720);
    }
}

function stopCamera() {
    if (camera != null) {
        camera.release();
        camera = null;
    }
}

function getMaxIndex(predArr) {
    let max = 0.0, idx = 0;
    for (let i = 0; i < predArr.length; i++) {
        if (predArr[i] > max) {
            max = predArr[i];
            idx = i;
        }
    }
    return idx;
}
