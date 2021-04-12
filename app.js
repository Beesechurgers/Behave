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
const fs = require('fs');
const path = require('path');
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);

var START_DETECTION = false;
var TESTED = false;
var camera = null;
var CAM_WIDTH = -1;
var CAM_HEIGHT = -1;
var FPS = 30;

const cascade = new cv.CascadeClassifier('haarcascade_frontalface_default.xml');
let model = null;
tf.loadLayersModel(`file://${__dirname}/model_tf/model.json`).then(m => {
    console.log("Model was loaded");
    model = m;
});

app.use(express.static('.'));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '\\index.html'))
});

app.get('/download', (req, res) => {
    res.download('video.mp4', 'video.mp4', (err) => {
        fs.unlinkSync('video.mp4');
        if (err) console.log("Err: ", err);
    });
});

let vw = null;
io.on('connection', (socket) => {
    console.log("Connected");
    socket.on('rec', (start) => {
        START_DETECTION = start;
        if (start) {
            startCamera();
        } else {
            if (vw != null) {
                vw.release();
                vw = null;
            }
            TESTED = false;
            stopCamera();
            io.emit('fix-img');
        }
    })

    socket.on('dn', (_) => {
        io.emit('dn-ret', vw == null && fs.existsSync('video.mp4'));
    });

    socket.on('disconnect', (_) => {
        console.log("Disconnected");
        if (fs.existsSync('video.mp4')) {
            fs.unlinkSync('video.mp4');
        }
    })
});

var emotion = "NA", fps = 0;
var gray = null, faces = null, values = new Float32Array(1), area = null, bitImg = null,
    pred = null, img = null;
let tfImage = null;
let roi, p1, p2, rect;
var count = 4;
const emots = ["angry", 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
setInterval(() => {
    if (!START_DETECTION) {
        return
    }
    var frame = camera.read().flip(1);
    fps++;
    gray = frame.cvtColor(cv.COLOR_BGR2GRAY);
    faces = cascade.detectMultiScale(gray, 1.3, 5);
    for (let i = 0; i < faces.numDetections.length; i++) {
        const ex = faces.objects[i];
        frame.drawRectangle(new cv.Point2(ex.x, ex.y), new cv.Point2(ex.x + ex.width, ex.y + ex.height), new cv.Vec3(0, 255, 0), 2);

        if (model != null) {
            (async () => {
                p1 = new cv.Point2(ex.x, ex.y);
                p2 = new cv.Point2(ex.x + ex.width, ex.y + ex.height);
                rect = new cv.Rect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
                roi = gray.getRegion(rect);
                area = roi.resize(48, 48);

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
    img = cv.imencode('.jpg', frame).toString('base64');
    if (TESTED) {
        if (vw != null) {
            vw.write(frame);
        }
        io.emit('image', img, CAM_WIDTH, CAM_HEIGHT);
    }
    img = null;
}, 1000 / 24);

setInterval(() => {
    if (!START_DETECTION) {
        return;
    }
    if (TESTED) {
        console.log('FPS', fps);
    } else {
        if (count == 0) {
            count = 4;
            TESTED = true;
            FPS = fps;
            if (vw == null) {
                vw = new cv.VideoWriter("video.mp4", -1, FPS, new cv.Size(CAM_WIDTH, CAM_HEIGHT), true);
                io.emit('test-done', 'done');
            }
        }
        count--;
    }
    fps = 0;
}, 1000);

server.listen(5000, () => {
    console.log('Listening on 5000');
});

function startCamera() {
    if (camera == null) {
        camera = new cv.VideoCapture(0);
        CAM_WIDTH = camera.get(cv.CAP_PROP_FRAME_WIDTH);
        CAM_HEIGHT = camera.get(cv.CAP_PROP_FRAME_HEIGHT);
    }
}

function stopCamera() {
    if (camera != null) {
        camera.release();
        camera = null;
        resetVars();
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

function resetVars() {
    emotion = "NA", fps = 0;
    gray = null, faces = null, values = new Float32Array(1), area = null, bitImg = null,
        pred = null, img = null;
    tfImage = null;
    roi = null, p1 = null, p2 = null, rect = null;
}
