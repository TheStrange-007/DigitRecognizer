const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
let isDrawing = false;
let x = 0;
let y = 0;
var offsetX;
var offsetY;
// get the canvas element and the save button element
const saveBtn = document.getElementById("saveBtn");
const predictBtn = document.getElementById("predictBtn");

function argmax(array) {
    if (array.length === 0) {
      return -1; // Return -1 for an empty array
    }
  
    let maxIndex = 0;
    let maxValue = array[0];
  
    for (let i = 1; i < array.length; i++) {
      if (array[i] > maxValue) {
        maxIndex = i;
        maxValue = array[i];
      }
    }
  
    return maxIndex;
}  

// add click event listener to the save button
predictBtn.addEventListener("click", () => {
    // create a new image element
    const img = new Image();
    // set the image source to the canvas data URL
    img.src = canvas.toDataURL("image/png");
    // send the image data to the Flask server using AJAX
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/predict_image", true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {

            console.log(xhr.responseText);

            let proba = JSON.parse(xhr.responseText); // Parsing to array

            var bars = document.querySelectorAll('.bars li .bar');
            for (let i=0; i<proba.length; i++)
            {
                var percentage = proba[i];
                bars[i].style.height = percentage + '%';
                bars[i].setAttribute('data-percentage', percentage);
                bars[i].style.transition = 'height 1s';
            }
            document.getElementById("prediction").innerHTML= `PREDICTED: ${argmax(proba)}`;
        }
    };
    xhr.send(JSON.stringify({
        "image": img.src
    }));
});

// add click event listener to the save button
saveBtn.addEventListener("click", () => {
    const img = new Image();
    const link = document.createElement("a");
    link.href = img.src;
    link.download = "drawing.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

function startup() {
    canvas.addEventListener('touchstart', handleStart);
    canvas.addEventListener('touchend', handleEnd);
    canvas.addEventListener('touchcancel', handleCancel);
    canvas.addEventListener('touchmove', handleMove);
    canvas.addEventListener('mousedown', (e) => {
        x = e.offsetX;
        y = e.offsetY;
        isDrawing = true;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDrawing) {
            drawLine(context, x, y, e.offsetX, e.offsetY);
            x = e.offsetX;
            y = e.offsetY;
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (isDrawing) {
            drawLine(context, x, y, e.offsetX, e.offsetY);
            x = 0;
            y = 0;
            isDrawing = false;
        }
    });
}

document.addEventListener("DOMContentLoaded", startup);

function handleStart(evt) {
    evt.preventDefault();
    const touches = evt.changedTouches;
    offsetX = canvas.getBoundingClientRect().left;
    offsetY = canvas.getBoundingClientRect().top;
}

function handleMove(evt) {
    evt.preventDefault();
    const touches = evt.changedTouches;
    const color = 'white';
    for (let i = 0; i < touches.length; i++) {
        context.beginPath();
        context.moveTo(touches[i].clientX - offsetX, touches[i].clientY - offsetY);
        context.lineTo(touches[i].clientX - offsetX + 1, touches[i].clientY - offsetY + 1);
        context.strokeStyle = color;
        context.lineWidth = 20;
        context.lineJoin = "round";
        context.closePath();
        context.stroke();
    }
}

function handleEnd(evt) {
    evt.preventDefault();
}

function handleCancel(evt) {
    evt.preventDefault();
}

function drawLine(context, x1, y1, x2, y2) {
    context.beginPath();
    context.strokeStyle = 'white';
    context.lineWidth = 20;
    context.lineJoin = "round";
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.closePath();
    context.stroke();
}

function clearArea() {
    context.clearRect(0, 0, canvas.width, canvas.height);
}
