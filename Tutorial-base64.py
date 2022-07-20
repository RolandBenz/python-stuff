from base64 import b64encode
import base64
import sys # to access the system
import cv2 #opencv-python
import io, numpy as np
from PIL import Image # pillow



def one_64encodeText():
    # 1: encode text
    # b makes it binary
    text_binary = b'Hey'
    print('1:\n', text_binary)
    # encode 64 (still binary)
    print('1:\n', b64encode(text_binary))


def two_64encodeImage_saveAsText():
    # 2. Encode image
    # open & read binary file in mode: rb (read binary)
    image = open('deer.gif', 'rb')
    image_read = image.read()
    image.close()
    # 64encode (still binary)
    image_64encoded = base64.urlsafe_b64encode(image_read)
    print('2:\n', image_64encoded)
    # write image_64encoded to new text file in mode: wb (write binary)
    encoded_result = open('results/deer_encoded.txt', 'wb')
    encoded_result.write(image_64encoded)
    encoded_result.close()


def three_decode_64encodedImage_savedAsText():
    # 3. decode image
    # open & read binary file in mode: rb (read binary)
    image_64encoded_txt = open('results/deer_encoded.txt', 'rb')
    image_64encoded_txt_read = image_64encoded_txt.read()
    print("3:\n", image_64encoded_txt_read)
    image_64encoded_txt.close()
    # decode 64 encoded image
    image_decoded = base64.urlsafe_b64decode(image_64encoded_txt_read)
    # write image_decoded to new gif file in mode: wb (write binary)
    image_decoded_result = open('results/deer_decoded.gif', 'wb')
    image_decoded_result.write(image_decoded)
    image_decoded_result.close()


def four_64encodeImage_decodeImage():
    # 4. Decode image directly from image_64_encoded
    image = open('deer.gif', 'rb')
    image_read = image.read()
    image.close()
    image_64encoded = base64.urlsafe_b64encode(image_read)
    image_decoded = base64.urlsafe_b64decode(image_64encoded)
    # write image_decoded to new gif file in mode: wb (write binary)
    image_decoded_result_ = open('results/deer_decoded_.gif', 'wb')
    image_decoded_result_.write(image_decoded)
    image_decoded_result_.close()
    # show image_decoded
    img = Image.open("results/deer_decoded_.gif")
    img.show()
    img.close()
    # show image_decoded in opencv-cv2
    img = cv2.imread("results/deer_decoded_.gif", cv2.IMREAD_ANYCOLOR)
    while True:
        cv2.imshow("Deer", img)
        cv2.waitKey(0)
        sys.exit()
    cv2.destroyAllWindows()  # destroy all windows


def five_decode_64encodedImage_savedAsString_andShowIt():
    # Image as 64encoded binary string
    image_64encoded_string = b'iVBORw0KGgoAAAANSUhEUgAAArUAAAK/AQAAAABfQAiKAAATrElEQVR4nO3dS47cthYAUAkaaKgdhBtJwm0ZeIFZb5Shl+CNPKArIw97Ca4gAw+7Ag9ajVaLD6L4F//3ynaAaGJ3t+oURZGXH1FS1/3Y23g5x52uJ7l3ZLBfxD9kRnYH6S7YLt/dFdkdpcvRXVHCJnz3avHo7sCRC/DIbzju6NaAgYufe7A7Bd1uTzbE3WrWz5a7/XzHcXtTqsbNHRYEd/HcpeuGtePQwLO7+ixNfN0y44LtEs5FKWPNroyI0xZh9tRteUB394bpMi42UdgA7mq5s6gL+3bpej4DXG5ctoiiu20PW8Rhc/t5s126andLb0cXiHuNugTgyio1vVfNg3D/4tvXkRXiio+Kloxsidzc+yDciQNcNh9dPgl3BLmrdqcNU+VMuM/tLhVtGNWugq9b0nHcvfUZtNtD8oEIjO3uxowrw3CnLdS6LjVue1wftxDTva26VRtXIt0O6N66/nUVZ2wWVW/EcAfOl27ku7uIEN8juZzzZ9vtMFxZXBftEtFcgF2ROP5ZNBsiN6R7E18J6O8w4zLpThiuKK10d7ciR7U7gNzJd+et7EkX0P8d9XnbYwWb9xgs/oLo8lmUkbs4EoDbc74S5V73BgTD7ez0SpcJl8Bcarm3rpfuLHtp7duk3S0Wi6BGhctgrjhxeyvH73sQJuKLOMztHXeS7rq5sHEsM+5W124ia1ZRUEDuZLtU9hz4Vjtgbm/ctVM9En4Z5dls37g8Q4TzQfYcOL9OYJca94mrztSNgl1i3L2HuuXMzMDu6LidqtuickDdS8gFDt/6b+GKEsdQ3C7mQofdbG8hTWRjurMK2qjvUlyXqbJFcFyyZ6Vw75YLneaapMs9F8g67q1T/4O7wzdwr8YFT6cOe7YSfa5GHLe33K5zO1cnuPBpcOOuyO5m/KTdASWcKfdXnac9ksu0O1suNEza7t1y4bPVbB99a3dAd2+WC5+1p8LlJ7irdK+Wa+/xU7vbG7c/uKzJJdq9GNcJO6wpU+REskkjpjta1iGcsaZCJ9zJstjBbap8Ygg4WSEMyRUTkiTh9m1BSLjMCjXUC78Q1w4J5OA2BfmtA9HbVXfEcUfO59GuYr7beM135Ga6y3NlvrZd81WuTtSgf5C/mkCuLqTa7UGuCA0PVgux/WKZLvpCz34los21I2Mv3CuSa31WuHck18pD7a4IrlVGlTsot6kz3B/Ga5TzdVqMO0JcK3ajulZbswVjLNc+NZiuXaW2GvhhhbrdoR+9uY9Irt2Ebe4nJNduyrec+YLk2l2PXl7VOMMVnRUEN/SbEVSPj/2xfcrkYrlNHbTj8JKiuId+kxosT3DXbcnPcifpqvaizfX7Y7KJvmK4bs/OcwmWO+wugbnkMByWoyO6qC9uYbPu+NCW3uk4fJfuDHa9EQ/l/Omm3cZRbdjld7k+AtMlwlWlmrXNTo1Hdx92qdLXOJsWcZfeuE0FbTiWh4Hzz2tvTcy0ZHDA7Tn/zAf5218ap2UCbme7FNFl/IH/Ln/L+No07RVyKX/gH4zbtLirD0zvEf7AH+XRb27LyDDsUv4EdLuAO1qdFc4X2ur6nzvLHVYElwXcN9O54nxudv2AZU/6sVY3cJV0sDptzelNu9vEfls5IxFXz4u3u37gttxtSV7zzJEfuAc5Rnov3D/bXS9wD/KaABNTgc/47j4V2DYzF2jgRjmm23onzW4gAKvLcfy2DzZQ3SvQDQRgotJ7Pw5zq1zvhBsXcpnkeB2PMuWOMNercKoQ8PsEuEzCDq48WT2fiRUxazfqp0hdg+v5TI/D3OLtMBIg0h34cpilqXSdgka1+3aYpanYJv9QuXbV/5q20Ts1g9IOc8MNrjcx57iNVw9771iJcqfDHG7dxg4TqrtLQNm7n3+TKHW2ZvBaE+JksMrVVa6NaV9bYFZVdOa06Q3o6lxkntvMOt0mvewenlxJXawvsTbIpW87CuynzejgBaXygEXhWAaEbFBFYDHfcVeXS2hjjNy30Trz+8HL3wDu9+msc3XXZUO7oBULftm6aRe4ssCrCxfjglgVuUwVk5X5b6A7Ouyi3a/ABU1uJbtq9xm6UMrLXh3UoAulnAy2giV0QZOdwYv1PdCFR33EBS88osadrR/BC4SsDL7v7leOsZBndNPLOP/EEdYP2iVNLmx6wHFNBq/mNCLc5m1VOe2+Ydw+bmLaBdU1/ZGbSvwbyu3jOsG6h/qGsDDRSvCM61pBjKC6qkisqtAtOK4uw3qRDNJzFZT7UbltM1FRV20vCAs/v427iiHsK7b7uuXtNsMFX6ja2cH9lXMuXJzHKmh3nx+Adx9897bHimekx3do9ypdpMeNHNKL5E6+i/TYFeP24txhpdc09i+cXwf+Fd3dguXAZ6TyYHeDb+NfaK7T+6M4/QexiSosOjr7FBeiKztQIqad5DI01x3HIbtflDvB7++wXW5cnLDuN3Dv0J7u445nZ7SnHPlzXFhPDRLufJL7YlyspxFNugaf4HJ81wrAd0x3ENVtn+o6XH4Auir+jie5PX9Ee1qZ1WCsI+Yzr7gdebCijh/QEB8G5wQ0xIe2UWvUgtZabBXjmfPf0cOOP/GHl8Fnue6MLeJT/M5ymR1/EV1qihmye04+uD0IvMjjDb3RXG8GHy1E+FfJsKqyx6JVOd89YUQkNqRYOfouUlE7usgzJsh1zp+Rwsrgo4tTggMuSgkOuCgZ4VdjrIwIuQgl7VDdkDL4WC1wXH98ITaEmhF0EUJa6LSd4z5juP51732DV4xgMUNwg8XsB3YDUeftCcENFjOUh6QEN2hAi7nQJvks99gY7xvCw47OcCPZ8LE+APeBZZOHbap2h09PNhzJhg9/1wb2yWkLYqWB17rUPSex0sBfKl3xocX98bh9qW3gvDvRw7FMPLm3zlXBK+NWp1e5F/dHfxsrXd2W6fsLgtuH2mcBD64bKbziknqVS81HYx0S8cfa+GAl8BItZHwN3M5XypplbIHtobK9cPLzFouQJqPKtp/cUrXUu+MD/90cxyR/mWKKXFGatPtu4l/4x7c8lXX3452tL6nfAv1JvaZ5uL7rfsVzn60/f0x8tNZtpZztGHbqzntsO1bjWOwDujjJPbrJWlS++WEnPAqp3/zkxmIq1MUwn47FF+WsyefEYReybUXeDT+5q/PODbyzdvO7sjiFNzTgboyJORbDDXZHELI32G2As8HhK0I2BHuR0Gy40nA2RAcLhVus6wStxBEWetpiXTJoSI+50GyIuOAWKNJDB4fesAtvMMPFDFp4Y9kLTm54YAWPDeEBEDgbIqUB3AKFXXhDHM5ecDZExu/whjjoIvQig+6P2gLFzhtCdy/oIvSfTjptp7mh8LDVts/nuE9npJdw/gh1Q3GH2Pe2Nm6hekH5Cg5ooQ414zO4xoVcfpa7voOyIbffZ/5gWyA+DHyFB55AeRj4c8EH693RrLpGdSd4LQ6et4kjtBcBlyAUh5BL4e4adhnnsKYo+FiyX7+K+wXQXfq6v98GtAVcxhHcQD0+0X2EdiHewu4d3OUJuATDDYyyznJR8iHs3oAjgTU0KoS7r+v7sHuBda1fgzczw93w6FjcSgQNleH0/utiuYF6IV5DAukAryzi3uFDzn+YG+r+Uv0GX2z3Chsbfj7JDVwQ+me6F5j7JXxTLIIbTO/+EGuAG5l+2B9CDlFfznKfgu4MdV+Ds5O7C+r4hWfPLie5Hch9SS9Pbu6gneW+plfVQBrOs9zUKtc298s+8ku4jTNSTzm3sYE7y33c3P/hu2K+NLVa/Qdz+fdymwMw5cvPZ7g8XR4gbjIfAIH9LDe5uLPdDU1qYLgvJ7nLSW560ShgpuC7uO0NXPrtiu3uuxTbfg15SbvNgTKz9PuHczNLv5tHLme5uaXf/zS3NfDk3NaedYZtrcjZOwx+MDd7J0Bj4DnJzd9h0Bgg8ndaVJNPZW51xfhQ5lZXjJPcu3mle3Krben3N1Lk3dqKcTnJDb2mAccNvMclsNVWOHXtI7dVVrit9/+e4bsiQJa88bnQfbXdkq2wpVdu8S1jhe5X+W/5LWNl7h/y3/JbxqrcilvGygLlf/d/Km69r3Ir7niscivuTCxz/zjXrbjjsaqBq7gz8V/36L4kyFrX/tica51b3VxrV+46cXLORaFy12kv7jm3PBs8N5PB5a7Tzt9y3YnyfHCgW6w7scDca/TJJNWu00+9hN1F155mN9g83/Ru5XHSrsZrpNlvcd1sDDbPN3NY5e2bl4+B8LZ2DW7vu8fxxq3FHbLu3hWpdSffPVTk2XaL+yXH5176CXb2K3ad0ySS5icYyx3cZe3e0yQKWbfayt6MnWLVgax03fClahOJu4UBggRdq7Iop9KNPJVR5/rq71g4DmBhV5dqnbxK12Gtgzz8QrllAa2Pub/4SJ07xFzvVW/GLetYe62v/af/8Dc7bXXvGTq8Xiq+51TjusUhNThTblGg9E5b0h3b3VSwUm5RQBu/jZs6RpW/RQNkUu5OjxUurXeLArDLJsumdgsCj9/HS9X9Gtfv66bc8fdyV0eHL/mP1Li6Fn8qcD8VHJTcdAbIhKfaLu3mA4+uxWuJq85bPkDo07ZM+aI5lLvmtVL7/5JVSbv5AGFewlLi6kzLujo6XEg+JaYOlbvyf8mcq3jPJtFHpt4LnNj6enfR7wUucrMVwzTdNP+B/rnavZ3kXmWgSLs6H7IFWLkqACUj1flutmIQvV9BAap3lzL3VcO5AqzfhtwXVFDLzbUYRJWbvuT4NJstaFOjm2sxiDqsoaT8NLuZ3Y2bazGIOr1jiWvOW65i1Lkv+gUzuC7XL9opdqdil5a7c1fr5iqy52aKO9MdgjL3rv5zlpup9rWunnvMhCnhjvkApdxLjTuUu7of/J1cai5DlLjrSe7SFTVD+95nuHt6WYE7yepW0rzt7r3cvXdlzZAQZU8uc2STrBbl7lzuXlWXGd8dK1xSEEnqXGrcTOSbuP3K8FwrwIrdkduvDC9zx0J31XEi53LzvuOMO1S5q0lvpiUcnCsWuV4Mr3ZphTt8R7c3ryIvdGUXvMC9/yBud6ZbsOut1mVlu161m+vd17sFswTV7sXup2XdW5Vb1P3trb5cgdu1uAWhr9NTQoUuKXf39jgTVJU7IbvqvI3lbldcHq5lLqtwe6uzUeqy7+3KW8JLXXqSS2pceoIrd8Z21QsPvrdb0tnQ7lhe385ySzobunwN5W5Jp6DTyy7yXfBa9y53Lhgy6M+UdjbK3LXKvdWml5W5V5mIs9zcQFa7tMy91Lqk0s0P/ff/TEXuqlNT6o75TsyJ7qI/Ver2edeEBZIdeFe6c7Xb1bgTtnuvd1k2qI51rlm1VuyOFW6+8XbcTN/TcTNBymj59y0Zd6pwhwp3LHGvTW4m+Bm3L3E7nYa8e9FupqxbEaTC7SrddJBy3EyZbHRZhctylWiy982UnW/g0gqX1rnpSu+4mbJT4xKDlbjq4EmFS3Jlx3Kn89x0pSfmZE11brpy1rhWoc2ORiw32xDVuNRJb/pcWG52lNPupitRjWtVhuyogTluuuy0u+my0+gOOZe5K9Yz++rz2ufOse3meuHfxk3nmT3Ll+vdfxs3fS7s0VWud2/nKatzk5XIc5PHZru5yum5mWNz3GTltOtCxnVGV7nK6bmpYxsqXKcuZN5k6ky5Zt7S2nv5m9rXcTOB3XdT+472VeNMYB8qXV3AM+7ouZl9TQHPVE7HzaxVcK4FZ9zJ+nPP071wd7ozvS9xsyzv6qPLu+rPQ2ZNwegsI0jva7tjiVs4Eji4iQq3uwpLu5Q7fcRkAR6d603pYEKtPevcdDCxXcLTyzvc5YJp154Z1euqItvgXDcmxS59Thfg3llPQZLBz74MS3m6oLnrYNJTPPaRsTdcV+3JxPs44oVH3idRMsXT26cqdz1frv+91ruPyQJ8cONlp7ePbP9YfGe9Xln/FC07Q8CN7qzXbefd8Yt1ZPVutOyMf1pHxp3jjLrybI2pXacbO7jR2kmcA0q7s7nEoVbeRwsEEe+iV27yWsM0E9td6UPata6QJRcKTIuT3mVzowVtm6owBzSlgqq9NGubn6SpCLEN/c16yClV1u2lZNqN5Zp2r9qNJ8GUHO3Gjo7wxbrkn1y5R133mgzB2jW3t8cOjdohgfMre0jsrd1Fu7HgR+2S/sovuxvJCKLWx5oHJsQKGrVD36Dc6GsyZ7EaUK8/yL62dXf6jDt9nLtFF4PkomVqh4SeX5zvOSZ47hYdmkpcVTFyrnpsi3Eje1LOYG7sPbPy1VX7T+pGvFj1HG+W+z7t2mvn+SW7pNUc/ftUxaCLfcsZ87qMoQ/o0sUeExVDufKZGHnX3KCQ7G0odz/yn2T7nHAHz41kGVnsyKPauNRkgU4kcz/obmxxIsS+d3LySA8cWKroaNea181OcF/1F0SPjS36JjFznEnWSXj0XDB554VJ41D6hAlihwB/m+Y+9ecCN1wgRuVWPF9Ou4kIPN6hbjhSDsqteM5eiat7ZQ3umr6btdWd1vS6cPHHh/b0xs6MqI9PLW56Pb/41ucGdyHJHBQR9bU9vdFGa/vbW319m1Z3POdve6SueI6hSk/G7dqKWTeumduQ6UnuCHRjFa5vCjvdsHa/pVvZ7HW3cHKW7G3prL40dF2Xd9u230puz2/ahPuE74qK8Vf5k2KrXPzkFt0k1bL1J7mioFU8EbN4yy+TatymU7IBYfs/jTMN0/JObwUAAAAASUVORK5CYII='
    # decode
    image_decoded = base64.urlsafe_b64decode(image_64encoded_string)
    print("5:\n", image_decoded)
    # open from bytestream instead of stored png
    # io.BytesIO() makes the trick
    img = Image.open(io.BytesIO(image_decoded))
    img.show()
    img.close()


if __name__ == '__main__':
    #
    if 0:
        one_64encodeText()
        two_64encodeImage_saveAsText()
        three_decode_64encodedImage_savedAsText()
        four_64encodeImage_decodeImage()
    else:
        five_decode_64encodedImage_savedAsString_andShowIt()

