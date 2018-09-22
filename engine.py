import math
import random as rnd
from PIL import Image, ImageDraw  # Подключим необходимые библиотеки.


def toLines(__points: list):
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


def toPerim(__lines: list):
    perim = 0
    if len(__lines) >= 2:
        perim += len(__lines[0])
        perim += len(__lines[len(__lines) - 1])
        for _ in range(1, len(__lines) - 1):
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
    return [__A[i] - __B[i] for i in range(0, min(len(__A), len(__B)))]


def delta(__A: list, __B: list):
    assert len(__A) == len(__B)
    s = sum([(__A[i] - __B[i]) ** 2 for i in range(0, min(len(__A), len(__B)))])
    return math.sqrt(s)


def toSumList(__A: list):
    return [toSum(x) for x in __A]


def toSum(__x):
    return __x


def Kmeans_init(__points: list, __count: int):
    unic = []
    for point in __points:
        if point not in unic:
            unic.append(point)

    clusterCenters = []
    for i in range(0, __count):
        if len(clusterCenters) == len(unic):
            raise AttributeError("unic values less then klusters")
        center = rnd.randint(0, len(unic) - 1)
        while center in clusterCenters:
            center = rnd.randint(0, len(unic) - 1)
        clusterCenters.append(center)
    return [unic[i] for i in clusterCenters]


def Kmeans_result_gen_init(__count: int):
    return {i: [] for i in range(0, __count)}


def Kmeans(__points: list, __count, __accur: float, __len_vector: int, __clusterCenters: list, __result: dict):
    unic = []
    for point in __points:
        if point not in unic:
            unic.append(point)

    flag = True

    res = __result.copy()
    while flag:
        oldClusters = [__clusterCenters[i] for i in range(0, __count)]
        for point in unic:
            deltas = []
            for i in range(0, __count):
                deltas.append(delta(__clusterCenters[i], point))
            minimDelta = min(deltas)

            kluster = deltas.index(minimDelta)
            __result[kluster].append(point)
        __clusterCenters.clear()
        for i in range(0, __count):
            temp = [0 for _ in range(0, __len_vector)]
            for j in range(0, len(__result[i])):
                for k in range(0, __len_vector):
                    temp[k] += __result[i][j][k]
            for j in range(0, __len_vector):
                if len(__result[i]) == 0:
                    temp[j] = 0
                else:
                    temp[j] = temp[j] / float(len(__result[i]))
            __clusterCenters.append(temp)
        deltasCenter = []
        for i in range(0, __count):
            deltasCenter.append(delta(oldClusters[i], __clusterCenters[i]))
        if max(deltasCenter) < __accur:
            flag = False
        else:
            for i in range(0, __count):
                __result[i] = res[i].copy()
    return __result


def binKmeans(__points: list, __accur: float):
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


def moment(__i, __j, __x_mean, __y_mean, __points):
    result = 0
    for point in __points:
        result += ((point[1] - __x_mean) ** __i) * ((point[0] - __y_mean) ** __j)
    return result


def toVector(__characteristics, __params: list):
    vector = []
    for param in __params:
        vector.append(__characteristics[param[0]] * param[1])
    return vector


def characteristics(__points: list):
    square = len(__points)
    lines = toLines(__points)
    perimeter = toPerim(lines)
    compact = (perimeter ** 2) / float(square)
    sum_x = 0
    sum_y = 0
    for point in __points:
        sum_y += point[0]
        sum_x += point[1]
    x_mean = sum_x / float(square)
    y_mean = sum_y / float(square)
    m02 = moment(0, 2, x_mean, y_mean, __points)
    m20 = moment(2, 0, x_mean, y_mean, __points)
    m11 = moment(1, 1, x_mean, y_mean, __points)
    elongation_up = (m20 + m02 + math.sqrt(((m20 - m02) ** 2) + (4 * (m11 ** 2))))
    elongation_down = (m20 + m02 - math.sqrt(((m20 - m02) ** 2) + (4 * (m11 ** 2))))
    elongation = 0
    if (elongation_down != 0) and (elongation_up != 0):
        elongation = float(elongation_up) / float(elongation_down)
    orientation = 0
    if (float(m20 - m02) != 0):
        orientation = (math.atan((2 * m11) / float(m20 - m02))) / float(2)
    return {"square": square, "perimeter": perimeter, "compact": compact, "mean_x": x_mean, "y_mean": y_mean,
            "elongation": elongation, "orientation": orientation}


def toBinImage(__input_name: str, __output_name: str):
    image = Image.open(__input_name)  # Открываем изображение.
    drawBin = ImageDraw.Draw(image)  # Создаем инструмент для рисования.

    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.

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

    image.save(__output_name, "PNG")


def toBinCheckMeans(__input_name: str, __output_name: str):
    imageOriginal = Image.open(__input_name)  # Открываем изображение.
    imageMap = Image.open(__input_name)  # Открываем изображение.

    drawMap = ImageDraw.Draw(imageMap)  # Создаем инструмент для рисования.

    width = imageOriginal.size[0]  # Определяем ширину.
    height = imageOriginal.size[1]  # Определяем высоту.

    pix = imageOriginal.load()  # Выгружаем значения пикселей.
    binMap = []
    for i in range(0, height):
        temp = []
        for j in range(0, width):
            if not checkMeans(pix, j, i, width, height):
                drawMap.point((j, i), (255, 255, 255))
                temp.append(0)
            else:
                drawMap.point((j, i), (0, 0, 0))
                temp.append(1)
        binMap.append(temp)
    imageMap.save(__output_name, "PNG")
    return binMap


def findObjectsOnImage(__input_name: str, binMap):
    image = Image.open(__input_name)

    width = image.size[0]
    height = image.size[1]
    obj = {}
    cur = 0

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
    return obj


def learnKMean(__input_name: str, __object_char, __obj, __count: int, __output_name: str, __params: list,
               __colors: list, __KMEANS, __KRESULT):
    image = Image.open(__input_name)
    draw = ImageDraw.Draw(image)

    width = image.size[0]
    height = image.size[1]

    for i in range(0, width):
        for j in range(0, height):
            draw.point((i, j), (255, 255, 255))
    klusterVectors = []
    for i in __object_char.keys():
        klusterVectors.append(toVector(__object_char[i], __params))
    if len(__KMEANS) == 0:
        __KMEANS = Kmeans_init(klusterVectors, __count)
        __KRESULT = Kmeans_result_gen_init(__count)
    result = Kmeans(klusterVectors, __count, 0.00003, len(__params), __KMEANS, __KRESULT)
    for i in __obj.keys():
        vector = toVector(__object_char[i], __params)
        for j in result.keys():
            if vector in result[j]:
                for value in __obj[i]:
                    draw.point((value[1], value[0]), __colors[j % len(__colors)])

    image.save(__output_name, "PNG")
    return __KMEANS, result


if __name__ == "__main__":
    KMEANS = []
    KRESULT = {}

    pictures = [("./resources/Laba_2_easy/P0001460.jpg", 6),  # 3),
                ("./resources/Laba_2_easy/P0001461.jpg", 6),
                ("./resources/Laba_2_easy/P0001468.jpg", 6),
                ("./resources/Laba_2_easy/P0001469.jpg", 6),
                ("./resources/Laba_2_easy/P0001471.jpg", 6),

                ("./resources/Laba_2_hard/P0001464.jpg", 6),
                ("./resources/Laba_2_hard/P0001465.jpg", 6),
                ("./resources/Laba_2_hard/P0001467.jpg", 6),
                ("./resources/Laba_2_hard/P0001470.jpg", 6),  # 5),
                ("./resources/Laba_2_hard/P0001472.jpg", 6)]
    for oper in range(0, len(pictures)):
        binName = "Bin" + str(oper) + ".png"
        print("to bin image ...")
        toBinImage(pictures[oper][0], binName)

        print("to bin check 4 means ...")
        binMap = toBinCheckMeans(binName, "RES_" + binName)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
                  (100, 0, 0), (0, 100, 0), (0, 0, 100), (100, 100, 0), (0, 100, 100), (100, 0, 100),
                  (175, 0, 0), (0, 175, 0), (0, 0, 175), (175, 175, 0), (0, 175, 175), (175, 0, 175)]

        print("find objects on image ...")
        obj = findObjectsOnImage("RES_" + binName, binMap)

        objectChar = {i: characteristics(obj[i]) for i in obj.keys()}
        print(objectChar)

        # params = ["square", "perimeter", "compact", "mean_x", "y_mean", "elongation", "orientation"]
        params = [("square", 4), ("perimeter", 0.8), ("compact", 2)]

        print("kmean ...")
        res = learnKMean("RES_" + binName, objectChar, obj, pictures[oper][1], "RES_MAP_" + binName, params, colors,
                         KMEANS, KRESULT)


        KMEANS = res[0]
        KRESULT = res[1]

        print("-------------------------------------------------")
        print("Result K:")
        for i in range(0, pictures[oper][1]):
            print("-----------------------------------------------")
            print(str(i))
            print("-----------------------------------------------")
            for point in KRESULT[i]:
                print(point)



