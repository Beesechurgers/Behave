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

const path = require('path');
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);

app.use(express.static('.'));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '\\index.html'))
});

io.on('connection', socket => {
    console.log("Connected");
});

setInterval(() => {
    io.emit('image', 'random data');
}, 1000);

server.listen(5000, () => {
    console.log('Listening on 5000');
});
