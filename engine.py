import math
import random as rnd

from PIL import Image, ImageDraw  # Подключим необходимые библиотеки.

def toLines(__points:list):
    heights = [i[0] for i in __points]
    minHeight = min(heights)
    maxHeight = max(heights)
    width = [i[1] for i in __points]
    minWidth = min(width)
    if minHeight == maxHeight:
        return []
    lines = [[] for _ in range(minHeight, maxHeight + 1)]
    for point in __points:
        lines[point[0] - minHeight].append(point[1] - minWidth)
    return lines

def toPerim(__lines:list):
    perim = 0
    if len(__lines) >= 2:
        perim += len(__lines[0])
        perim += len(__lines[len(__lines) - 1])
        for i in range(1, len(__lines) - 1):
            if len(__lines) >= 2:
                perim += 2
    return perim

def toGray(call_pixel: tuple):
    assert len(call_pixel) == 3
    return 0.3 * call_pixel[0] + 0.59 * call_pixel[1] + 0.11 * call_pixel[2]


def bc(call_pixel: tuple):
    return not ((call_pixel[0] == 255) or (call_pixel[1] == 255) or (call_pixel[2] == 255))


def checkMeans(__pixelMap, __x: int, __y: int, __width: int, __height: int):
    if __x == 0:
        if __y == 0:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x + 1, __y])) and (
                        bc(__pixelMap[__x, __y + 1])))
        elif __y >= __height - 1:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x + 1, __y])) and (
                        bc(__pixelMap[__x, __y - 1])))
        else:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x + 1, __y])) and (
                        bc(__pixelMap[__x, __y + 1])) and (
                        bc(__pixelMap[__x, __y - 1])))
    elif __x >= __width - 1:
        if __y == 0:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x - 1, __y])) and (
                        bc(__pixelMap[__x, __y + 1])))
        elif __y >= __height - 1:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x - 1, __y])) and (
                        bc(__pixelMap[__x, __y - 1])))
        else:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x - 1, __y])) and (
                        bc(__pixelMap[__x, __y + 1])) and (
                        bc(__pixelMap[__x, __y - 1])))
    else:
        if __y == 0:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x - 1, __y])) and (
                        bc(__pixelMap[__x + 1, __y])) and (
                        bc(__pixelMap[__x, __y + 1])))
        elif __y >= __height - 1:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x - 1, __y])) and (
                        bc(__pixelMap[__x + 1, __y])) and (
                        bc(__pixelMap[__x, __y - 1])))
        else:
            return (bc(__pixelMap[__x, __y]) and (
                bc(__pixelMap[__x - 1, __y])) and (
                        bc(__pixelMap[__x + 1, __y])) and (
                        bc(__pixelMap[__x, __y + 1])) and (
                        bc(__pixelMap[__x, __y - 1])))


def toBredli(__picture_name: str, __t: float, __resultName: str):
    image = Image.open(__picture_name)  # Открываем изображение.
    imageBin = Image.open(__picture_name)  # Открываем изображение.
    drawBin = ImageDraw.Draw(imageBin)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    SCoef = width // 8
    SCoef_half = SCoef // 2

    integral_image = [[0 for _ in range(0, width)] for _ in range(0, height)]
    for i in range(0, width):
        current_sum = 0
        for j in range(0, height):
            SCoef = toGray(pix[i, j])
            current_sum += SCoef
            if i == 0:
                integral_image[j][i] = current_sum
            else:
                integral_image[j][i] = integral_image[j][i - 1] + current_sum
    for i in range(0, width):
        for j in range(0, height):
            x1 = i - SCoef_half
            x2 = i + SCoef_half
            y1 = j - SCoef_half
            y2 = j + SCoef_half
            if x1 < 0:
                x1 = 0
            if x2 >= width:
                x2 = width - 1
            if y1 < 0:
                y1 = 0
            if y2 >= height:
                y2 = height - 1
            count = (x2 - x1) * (y2 - y1)
            current_sum = integral_image[y2][x2] - integral_image[y1][x2] - \
                          integral_image[y2][x1] + integral_image[y1][x1]
            if (toGray(pix[i, j]) * count) < (current_sum * (1.0 - __t)):
                drawBin.point((i, j), (255, 255, 255))
            else:
                drawBin.point((i, j), (0, 0, 0))

    imageBin.save(__resultName, "PNG")


def subList(__A: list, __B: list):
    assert len(__A) == len(__B)
    return [__A[i] - __B[i] for i in range(0, min(len(A), len(B)))]


def delta(__A: list, __B: list):
    assert len(__A) == len(__B)
    s = sum([(__A[i] - __B[i]) ** 2 for i in range(0, min(len(A), len(B)))])
    return math.sqrt(s)

def toSumList(__A:list):
    return [toSum(x) for x in __A]

def toSum(__x):
    return __x


def Kmeans(__points: list, __count, __accur: float):
    unic = []
    for point in __points:
        if point not in unic:
            unic.append(point)

    clusterCenters = []
    result = {i: [] for i in range(0, __count)}
    for i in range(0, __count):
        center = rnd.randint(0, len(unic))
        while center in clusterCenters:
            center = rnd.randint(0, len(unic))
        clusterCenters.append(center)

    clusterCenters = [unic[i] for i in clusterCenters]
    flag = True

    while flag:
        oldClusters = [clusterCenters[i] for i in range(0, __count)]
        for point in unic:
            deltas = []
            for i in range(0, __count):
                deltas.append(delta(clusterCenters[i], point))
            minimDelta = min(deltas)

            kluster = deltas.index(minimDelta)
            result[kluster].append(point)
        clusterCenters.clear()
        for i in range(0, __count):
            sumValues = [sum(result[i][j]) / float(len(result[i])) for j in range(0, len(result[i]))]
            clusterCenters.append([sumValues])
        deltasCenter = []
        for i in range(0, __count):
            deltasCenter.append(delta(oldClusters[i], clusterCenters[i]))
        if max(deltasCenter) < __accur:
            flag = False
        else:
            for i in range(0, __count):
                result[i].clear()
    return result



def binKmeans(__points: list, __accur):
    unic = []
    for point in __points:
        if point[0] not in unic:
            unic.append(point[0])

    A = rnd.randint(0, len(unic))
    B = rnd.randint(0, len(unic))
    while B == A:
        B = rnd.randint(0, len(unic))

    AValues = []
    BValues = []
    flag = True

    while flag:
        Aold = A
        Bold = B
        AValues.clear()
        BValues.clear()
        for point in unic:
            distA = abs(A - point)
            distB = abs(B - point)
            if distA <= distB:
                AValues.append(point)
            else:
                BValues.append(point)

        if (len(AValues)) == 0 or (len(BValues) == 0):
            break
        A = sum(AValues) / float(len(AValues))
        B = sum(BValues) / float(len(BValues))
        if (abs(A - Aold) <= __accur) and (abs(B - Bold) <= __accur):
            flag = False

    binAValue = 1
    binBValue = 0

    if A >= B:
        binAValue = 0
        binBValue = 1

    return [(binAValue if i[0] in AValues else binBValue, i[1], i[2]) for i in __points]


if __name__ == "__main__":
    pictures = ["./resources/Laba_2_easy/P0001460.jpg",
                # "./resources/Laba_2_easy/P0001461.jpg",
                # "./resources/Laba_2_easy/P0001468.jpg",
                # "./resources/Laba_2_easy/P0001469.jpg",
                "./resources/Laba_2_easy/P0001471.jpg",

                # "./resources/Laba_2_hard/P0001464.jpg",
                # "./resources/Laba_2_hard/P0001465.jpg",
                # "./resources/Laba_2_hard/P0001467.jpg",
                "./resources/Laba_2_hard/P0001470.jpg",
                "./resources/Laba_2_hard/P0001472.jpg"]
    for oper in range(0, len(pictures)):
        resultName = "Bredli" + str(oper) + ".png"
        binName = "Bin" + str(oper) + ".png"
        # toBredli(pictures[oper], 0.1, resultName)

        imageOriginal = Image.open(pictures[oper])  # Открываем изображение.
        image = Image.open(pictures[oper])  # Открываем изображение.
        imageBin = Image.open(pictures[oper])  # Открываем изображение.
        drawBin = ImageDraw.Draw(imageBin)  # Создаем инструмент для рисования.

        width = imageOriginal.size[0]  # Определяем ширину.
        height = imageOriginal.size[1]  # Определяем высоту.
        pix = imageOriginal.load()  # Выгружаем значения пикселей.

        kPix = []

        for i in range(0, width):
            for j in range(0, height):
                kPix.append((int(round(toGray(pix[i, j]))), i, j))

        binPix = binKmeans(kPix, 0.0000003)

        for bpix in binPix:
            if bpix[0] == 1:
                drawBin.point((bpix[1], bpix[2]), (255, 255, 255))
            else:
                drawBin.point((bpix[1], bpix[2]), (0, 0, 0))

        imageBin.save(binName, "PNG")
        print("BIN COMPLETE = " + str(binName))
        resultName = binName

        imageBredli = Image.open("./" + resultName)  # Открываем изображение.
        imageMap = Image.open("./" + resultName)  # Открываем изображение.

        drawBredli = ImageDraw.Draw(imageBredli)  # Создаем инструмент для рисования.
        drawMap = ImageDraw.Draw(imageMap)  # Создаем инструмент для рисования.

        width = imageBredli.size[0]  # Определяем ширину.
        height = imageBredli.size[1]  # Определяем высоту.

        pix = imageBredli.load()  # Выгружаем значения пикселей.

        for i in range(0, width):
            for j in range(0, height):
                if not checkMeans(pix, i, j, width, height):
                    drawMap.point((i, j), (255, 255, 255))
                else:
                    drawMap.point((i, j), (0, 0, 0))

        imageMap.save("RES_" + resultName, "PNG")

        imageMap = Image.open("./" + resultName)  # Открываем изображение.
        imageMapRes = Image.open("./" + resultName)  # Открываем изображение.
        drawMap = ImageDraw.Draw(imageMap)  # Создаем инструмент для рисования.
        drawMapRes = ImageDraw.Draw(imageMapRes)  # Создаем инструмент для рисования.
        width = imageMap.size[0]  # Определяем ширину.
        height = imageMap.size[1]  # Определяем высоту.

        pix = imageMap.load()  # Выгружаем значения пикселей.

        binMap = []
        for i in range(0, height):
            temp = []
            for j in range(0, width):
                if checkMeans(pix, j, i, width, height):
                    temp.append(1)
                else:
                    temp.append(0)
            binMap.append(temp)

        cur = 0

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
                  (100, 0, 0), (0, 100, 0), (0, 0, 100), (100, 100, 0), (0, 100, 100), (100, 0, 100),
                  (175, 0, 0), (0, 175, 0), (0, 0, 175), (175, 175, 0), (0, 175, 175), (175, 0, 175)]

        obj = {}
        for i in range(0, height):
            for j in range(0, width):
                kn = j - 1
                if kn <= 0:
                    kn = 1
                    B = 0
                else:
                    B = binMap[i][kn]
                km = i - 1
                if km <= 0:
                    km = 1
                    C = 0
                else:
                    C = binMap[km][j]
                A = binMap[i][j]
                if not A:
                    pass
                elif (B == 0) and (C == 0):
                    if len(obj.keys()) == 0:
                        cur = 0
                    else:
                        cur = min(obj.keys())
                        while cur in obj.keys():
                            cur += 1
                    binMap[i][j] = cur
                    if cur not in obj.keys():
                        obj[cur] = []
                    obj[cur].append((i, j))
                elif (not (B == 0)) and (C == 0):
                    binMap[i][j] = B
                    if B not in obj.keys():
                        obj[B] = []
                    obj[B].append((i, j))
                elif (B == 0) and (not (C == 0)):
                    binMap[i][j] = C
                    if C not in obj.keys():
                        obj[C] = []
                    obj[C].append((i, j))
                elif (not (B == 0)) and (not (C == 0)):
                    binMap[i][j] = B
                    if B not in obj:
                        obj[B] = []
                    obj[B].append((i, j))
                    if B != C:
                        if C in obj.keys():
                            for i_1 in obj[C]:
                                binMap[i_1[0]][i_1[1]] = B
                            obj[B] = obj[B] + obj[C]
                            del obj[C]

        for i in range(0, width):
            for j in range(0, height):
                drawMapRes.point((i, j), (255, 255, 255))
        objectChar = {i: {"square": len(obj[i]), "perim": toPerim(toLines(obj[i]))} for i in obj.keys()}
        print(objectChar)
        for key in obj.keys():
            for value in obj[key]:
                drawMapRes.point((value[1], value[0]), colors[key % len(colors)])

        imageMapRes.save("RES_MAP_" + resultName, "PNG")

        print("MAP_COMPLETE " + str(resultName))