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
const path = require('path');
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);

const camera = new cv.VideoCapture(0);
camera.set(cv.CAP_PROP_FRAME_WIDTH, 600);
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 400);

const cascade = new cv.CascadeClassifier('haarcascade_frontalface_default.xml');
const FPS = 24;

app.use(express.static('.'));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '\\index.html'))
});

io.on('connection', socket => {
    console.log("Connected");
    socket.on('image', (data) => {
        console.log(data);
    });
});

setInterval(() => {
    var frame = camera.read().flip(1);
    gray = frame.cvtColor(cv.COLOR_BGR2GRAY);
    faces = cascade.detectMultiScale(gray, 1.3, 5);
    for (let i = 0; i < faces.numDetections.length; i++) {
        const ex = faces.objects[i];
        frame.drawRectangle(new cv.Point2(ex.x, ex.y), new cv.Point2(ex.x + ex.width, ex.y + ex.height), new cv.Vec3(0, 255, 0), 2);
    }
    const img = cv.imencode('.jpg', frame).toString('base64');
    io.emit('image', img);
}, 1000 / FPS);

server.listen(5000, () => {
    console.log('Listening on 5000');
});
