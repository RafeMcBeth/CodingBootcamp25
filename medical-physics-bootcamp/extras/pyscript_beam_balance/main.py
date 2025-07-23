from js import document, window
import math, random

canvas = document.getElementById("canvas")
ctx = canvas.getContext("2d")
w, h = canvas.width, canvas.height

beam_x = w // 2
healthy_zone = (100, 300)  # x-range considered “safe”
dose = 0

def draw():
    global beam_x, dose
    ctx.clearRect(0, 0, w, h)

    # draw healthy zone
    ctx.fillStyle = "#144"
    ctx.fillRect(*healthy_zone, 0, h)
    ctx.fillRect(healthy_zone[0], 0, healthy_zone[1]-healthy_zone[0], h)

    # draw beam
    ctx.fillStyle = "#f55"
    ctx.fillRect(beam_x-5, 0, 10, h)

    # update dose
    if healthy_zone[0] <= beam_x <= healthy_zone[1]:
        dose = max(dose-0.01, 0)
    else:
        dose += 0.05

    # text
    ctx.fillStyle = "#eee"
    ctx.fillText(f"Dose penalty: {dose:.1f}", 10, 20)

def loop(_):
    draw()
    window.requestAnimationFrame(loop)

def on_key(ev):
    global beam_x
    if ev.key == "ArrowLeft":  beam_x = max(5, beam_x-10)
    if ev.key == "ArrowRight": beam_x = min(w-5, beam_x+10)

window.addEventListener("keydown", on_key)
window.requestAnimationFrame(loop)
