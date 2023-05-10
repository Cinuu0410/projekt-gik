# Opis programu:
# Program umożliwia edycję zdjęć poprzez nakładanie różnych filtrów. Można wybrać czy działamy na 1 zdjęciu czy na 2.
# Po wybraniu danej opcji otwiera się konkretne okienko, które pokazuje zdjęcie bazowe oraz zdjęcie po przeróbce.
# Wychodzi się z okienka poprzez przycisk "Wyjdź". Zaimplementowanych jest kilkadziesiąt metod
# umożliwiających nakładanie filtrów na zdjęcie
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

print("Na ile obrazkach chcesz pracować?:")
print("1 - praca na jednym obrazku\n2 - praca na dwóch obrazkach")
ile = input()


def apply_filter(filtr):
    global image
    img = Image.open("obraz.jpg")
    if filtr == "Rozjaśnianie liniowe":
        image = brightening(img)
    elif filtr == "Przyciemnianie liniowe":
        image = dimming(img)
    elif filtr == "Negatyw liniowo":
        image = negative(img)
    elif filtr == "Rozjaśnianie potęgowe":
        image = brighteningPotegowe(img)
    elif filtr == "Przyciemnanie potęgowe":
        image = dimmingPotegowe(img)
    elif filtr == "Histogram":
        image = histogram(img)
    elif filtr == "Kontrast obrazu barwnego":
        image = kontrastobrazubarwnego(img)
    elif filtr == "Roberts (poziomy)":
        image = robertspoziomy(img)
    elif filtr == "Roberts (pionowy)":
        image = robertspionowy(img)
    elif filtr == "Prewitt (poziomy)":
        image = prewittapoziomy(img)
    elif filtr == "Prewitt (pionowy)":
        image = prewittapionowy(img)
    elif filtr == "Sobel (poziomy)":
        image = sobelapoziomy(img)
    elif filtr == "Sobel (pionowy)":
        image = sobelapionowy(img)
    elif filtr == "Laplace":
        image = laplace(img)
    elif filtr == "Minimum":
        image = minimum(img)
    elif filtr == "Maksimum":
        image = maksimum(img)
    elif filtr == "Medianowy":
        image = medianowy(img)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")


def apply_filter2(filtr):
    global image
    img1 = Image.open("obraz.jpg")
    img2 = Image.open("obraz2.jpg")
    if filtr == "Suma":
        image = suma(img1, img2)
    elif filtr == "Odejmowanie":
        image = odejmowanie(img1, img2)
    elif filtr == "Różnica":
        image = roznica(img1, img2)
    elif filtr == "Mnożenie":
        image = mnozenie(img1, img2)
    elif filtr == "Mnożenie odwrotności":
        image = mnozenieOdwrotnosci(img1, img2)
    elif filtr == "Negacja":
        image = negacja(img1, img2)
    elif filtr == "Ciemniejsze":
        image = ciemniejsze(img1, img2)
    elif filtr == "Jaśniejsze":
        image = jasniejsze(img1, img2)
    elif filtr == "Wyłączenie":
        image = wylaczanie(img1, img2)
    elif filtr == "Nakładka":
        image = nakladka(img1, img2)
    elif filtr == "Ostre światło":
        image = ostreswiatlo(img1, img2)
    elif filtr == "Łagodne światło":
        image = lagodneswiatlo(img1, img2)
    elif filtr == "Rozcieńczenie":
        image = rozcienczanie(img1, img2)
    elif filtr == "Wypalanie":
        image = wypalanie(img1, img2)
    elif filtr == "Reflect mode":
        image = reflect(img1, img2)
    elif filtr == "Przezroczystość":
        image = przezroczystosc(img1, img2)
    image = ImageTk.PhotoImage(image)
    canvas1.image = image
    canvas1.create_image(0, 0, image=image, anchor="nw")


def brightening(img):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r = 1 * r + 127
            g = 1 * g + 127
            b = 1 * b + 127
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    return image_result


def dimming(img):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r = 1 * r - 100
            g = 1 * g - 100
            b = 1 * b - 100
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    return image_result


def negative(img):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r = 255 - r
            g = 255 - g
            b = 255 - b
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    return image_result


def brighteningPotegowe(img):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r = int(255 * ((r / 255) ** 0.33))
            g = int(255 * ((g / 255) ** 0.33))
            b = int(255 * ((b / 255) ** 0.33))
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    return image_result


def dimmingPotegowe(img):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r = int(255 * ((r / 255) ** 2))
            g = int(255 * ((g / 255) ** 2))
            b = int(255 * ((b / 255) ** 2))
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    return image_result


def histogram(img):
    histogram = img.histogram()

    plt.hist(histogram, bins=256, color="black")
    plt.title("Histogram")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba pikseli")
    plt.show()
    return histogram


def kontrastobrazubarwnego(img, c=70):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r = int((127 / (127 - c)) * (r - c))
            g = int((127 / (127 - c)) * (g - c))
            b = int((127 / (127 - c)) * (b - c))
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    return image_result


def robertspoziomy(img):
    plt.axis("off")
    plt.imshow(img)

    matrix = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r = sum(img.getpixel((i + x, j + y))[0] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            g = sum(img.getpixel((i + x, j + y))[1] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            b = sum(img.getpixel((i + x, j + y))[2] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            image_result.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def robertspionowy(img):
    plt.axis("off")
    plt.imshow(img)

    matrix = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r = sum(img.getpixel((i + x, j + y))[0] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            g = sum(img.getpixel((i + x, j + y))[1] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            b = sum(img.getpixel((i + x, j + y))[2] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            image_result.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def prewittapoziomy(img):
    plt.axis("off")
    plt.imshow(img)

    matrix = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r = sum(img.getpixel((i + x, j + y))[0] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            g = sum(img.getpixel((i + x, j + y))[1] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            b = sum(img.getpixel((i + x, j + y))[2] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            image_result.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def prewittapionowy(img):
    plt.axis("off")
    plt.imshow(img)

    matrix = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r = sum(img.getpixel((i + x, j + y))[0] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            g = sum(img.getpixel((i + x, j + y))[1] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            b = sum(img.getpixel((i + x, j + y))[2] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            image_result.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def sobelapoziomy(img):
    plt.axis("off")
    plt.imshow(img)

    matrix = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r = sum(img.getpixel((i + x, j + y))[0] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            g = sum(img.getpixel((i + x, j + y))[1] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            b = sum(img.getpixel((i + x, j + y))[2] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            image_result.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def sobelapionowy(img):
    plt.axis("off")
    plt.imshow(img)

    matrix = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r = sum(img.getpixel((i + x, j + y))[0] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            g = sum(img.getpixel((i + x, j + y))[1] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            b = sum(img.getpixel((i + x, j + y))[2] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            image_result.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def laplace(img):
    plt.axis("off")
    plt.imshow(img)
    matrix = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r = sum(img.getpixel((i + x, j + y))[0] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            g = sum(img.getpixel((i + x, j + y))[1] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            b = sum(img.getpixel((i + x, j + y))[2] * matrix[x][y] for x in range(-1, 2) for y in range(-1, 2))
            image_result.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def minimum(img, size=1):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            min_r, min_g, min_b = 255, 255, 255
            for x in range(max(0, i - size), min(w, i + size + 1)):
                for y in range(max(0, j - size), min(h, j + size + 1)):
                    r, g, b = img.getpixel((x, y))
                    min_r = min(min_r, r)
                    min_g = min(min_g, g)
                    min_b = min(min_b, b)
            image_result.putpixel((i, j), (min_r, min_g, min_b))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def maksimum(img, size=1):
    plt.axis("off")
    plt.imshow(img)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            max_r, max_g, max_b = 0, 0, 0
            for x in range(max(0, i - size), min(w, i + size + 1)):
                for y in range(max(0, j - size), min(h, j + size + 1)):
                    r, g, b = img.getpixel((x, y))
                    max_r = max(max_r, r)
                    max_g = max(max_g, g)
                    max_b = max(max_b, b)
            image_result.putpixel((i, j), (max_r, max_g, max_b))

    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def medianowy(img, size=3):
    plt.axis("off")
    plt.imshow(img)
    image_result = Image.new("RGB", (img.width, img.height))
    w, h = img.size
    for i in range(w):
        for j in range(h):
            pixels = []
            for x in range(max(0, i - size), min(w, i + size + 1)):
                for y in range(max(0, j - size), min(h, j + size + 1)):
                    pixels.append(img.getpixel((x, y)))
            pixels.sort()
            median_r, median_g, median_b = pixels[len(pixels) // 2]
            image_result.putpixel((i, j), (median_r, median_g, median_b))
    plt.axis("off")
    plt.imshow(image_result)
    return image_result


def suma(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = r + r1
            g = g + g1
            b = b + b1
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def odejmowanie(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = max(r - r1, 0)
            g = max(g - g1, 0)
            b = max(b - b1, 0)
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def roznica(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = abs(r - r1)
            g = abs(g - g1)
            b = abs(b - b1)
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def mnozenie(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = r * r1
            g = g * g1
            b = b * b1
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def mnozenieOdwrotnosci(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = 255 - ((255 - r) * (255 - r1) // 255)
            g = 255 - ((255 - g) * (255 - g1) // 255)
            b = 255 - ((255 - b) * (255 - b1) // 255)
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def negacja(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = 255 - abs(255 - r - r1)
            g = 255 - abs(255 - g - g1)
            b = 255 - abs(255 - b - b1)
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def ciemniejsze(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = min(r, r1)
            g = min(g, g1)
            b = min(b, b1)
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def jasniejsze(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = max(r, r1)
            g = max(g, g1)
            b = max(b, b1)
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def wylaczanie(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = r + r1 - 2 * r * r1 // 255
            g = g + g1 - 2 * g * g1 // 255
            b = b + b1 - 2 * b * b1 // 255
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def nakladka(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = r * (255 - 2 * g1) + 2 * r1 * g // 255
            g = g * (255 - 2 * b1) + 2 * g1 * b // 255
            b = b * (255 - 2 * r1) + 2 * b1 * r // 255
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def ostreswiatlo(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = 2 * b1 * r // 255 + (255 - 2 * b1) * r1 // 255 if r1 <= 127 else 2 * (r + b1) - 255
            g = 2 * b1 * g // 255 + (255 - 2 * b1) * g1 // 255 if g1 <= 127 else 2 * (g + b1) - 255
            b = 2 * b1 * b // 255 + (255 - 2 * b1) * b1 // 255 if b1 <= 127 else 2 * (b + b1) - 255
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def lagodneswiatlo(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = ((255 - r) * r1 * r1 // 255 + 2 * r * r1 * (255 - r1) // 255 + r * (255 - r1) * (
                    255 - r1) // 255) // 255
            g = ((255 - g) * g1 * g1 // 255 + 2 * g * g1 * (255 - g1) // 255 + g * (255 - g1) * (
                    255 - g1) // 255) // 255
            b = ((255 - b) * b * b // 255 + 2 * b * b1 * (255 - b1) // 255 + b * (255 - b1) * (
                    255 - b1) // 255) // 255
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def rozcienczanie(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = int(r / (255 - r1 + 1) * 255)
            g = int(g / (255 - g1 + 1) * 255)
            b = int(b / (255 - b1 + 1) * 255)
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def wypalanie(img, img2):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = 255 - min(255, (255 - r) * 255 // (r1 + 1))
            g = 255 - min(255, (255 - g) * 255 // (g1 + 1))
            b = 255 - min(255, (255 - b) * 255 // (b1 + 1))
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def reflect(img, img2, epsilon=0.001):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = int((r ** 2) / (1 - (r1 / 255 + epsilon)))
            g = int((g ** 2) / (1 - (g1 / 255 + epsilon)))
            b = int((b ** 2) / (1 - (b1 / 255 + epsilon)))
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


def przezroczystosc(img, img2, alpha=0.5):
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)

    image_result = Image.new("RGB", (img.width, img.height))

    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r1, g1, b1 = img2.getpixel((i, j))
            r = int(r * alpha + r1 * (1 - alpha))
            g = int(g * alpha + g1 * (1 - alpha))
            b = int(b * alpha + b1 * (1 - alpha))
            image_result.putpixel((i, j), (r, g, b))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(img2)
    return image_result


if ile == "1":
    ramka = tk.Tk()
    ramka.geometry("900x600")
    ramka.title("Ulepsz swój obrazek")
    ramka.config(bg="white")

    left = tk.Frame(ramka, width=200, height=500, bg="gray")
    left.pack(side="left", fill="y")

    canvas = tk.Canvas(ramka, width=700, height=500)
    img = ImageTk.PhotoImage(Image.open("obraz.jpg"))
    canvas.create_image(0, 0, image=img, anchor="nw")
    canvas.pack()

    filtr_label = tk.Label(left, text="Wybierz filtr", bg="white")
    filtr_label.pack()
    filtr_lista = ttk.Combobox(left, height=30,
                               values=["Rozjaśnianie liniowe", "Przyciemnianie liniowe", "Negatyw liniowo",
                                       "Rozjaśnianie potęgowe", "Przyciemnanie potęgowe", "Histogram",
                                       "Kontrast obrazu barwnego", "Roberts (poziomy)", "Roberts (pionowy)",
                                       "Prewitt (poziomy)", "Prewitt (pionowy)", "Sobel (poziomy)", "Sobel (pionowy)",
                                       "Laplace", "Minimum", "Maksimum", "Medianowy"])
    filtr_lista.pack()
    filtr_lista.bind("<<ComboboxSelected>>", lambda event: apply_filter(filtr_lista.get()))

    exit = tk.Button(left, text="Wyjdź", command=ramka.destroy)
    exit.pack(pady=20)

    ramka.mainloop()

if ile == "2":
    ramka = tk.Tk()
    ramka.attributes("-fullscreen", True)
    ramka.title("Ulepsz swój obrazek")
    ramka.config(bg="white")

    left = tk.Frame(ramka, width=200, height=500, bg="gray")
    left.pack(side="left", fill="y")

    ramka1 = tk.Frame(ramka)
    ramka1.pack(side="top")
    canvas1 = tk.Canvas(ramka1, width=700, height=500)
    canvas1.pack(side="top", anchor="nw")

    ramka2 = tk.Frame(ramka)
    ramka2.pack(side="right")
    canvas2 = tk.Canvas(ramka2, width=700, height=500)
    img2 = ImageTk.PhotoImage(Image.open("obraz2.jpg"))
    canvas2.create_image(0, 0, image=img2, anchor="nw")
    canvas2.pack(side="right", anchor="nw")

    ramka3 = tk.Frame(ramka)
    ramka3.pack(side="left")
    canvas3 = tk.Canvas(ramka3, width=700, height=500)
    img3 = ImageTk.PhotoImage(Image.open("obraz.jpg"))
    canvas3.create_image(350, 0, image=img3, anchor="n")
    canvas3.pack(side="top", anchor="nw")

    filtr_label = tk.Label(left, text="Wybierz filtr", bg="white")
    filtr_label.pack()
    filtr_lista = ttk.Combobox(left, height=30,
                               values=["Suma", "Odejmowanie", "Różnica", "Mnożenie", "Mnożenie odwrotności",
                                       "Negacja", "Ciemniejsze", "Jaśniejsze", "Wyłączenie", "Nakładka",
                                       "Ostre światło", "Łagodne światło", "Rozcieńczenie", "Wypalanie",
                                       "Reflect mode", "Przezroczystość"])
    filtr_lista.pack()
    filtr_lista.bind("<<ComboboxSelected>>", lambda event: apply_filter2(filtr_lista.get()))
    exit = tk.Button(left, text="Wyjdź", command=ramka.destroy)
    exit.pack(pady=20)
    ramka.mainloop()
