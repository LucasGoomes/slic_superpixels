import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color
import streamlit as st
from slic import SLIC 

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def main():
    st.title('Segmentação de Imagem com SLIC')

    # Carregar a imagem
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = bgr_to_rgb(img)
        st.image(img_rgb, caption='Imagem Original', use_column_width=True)

        # Converter para LAB
        img_CIElab = color.rgb2lab(img)

        # Parâmetros do SLIC
        k = st.slider("Número de Clusters", min_value=10, max_value=300, value=100, step=10)
        iters = st.slider("Número de Iterações", min_value=1, max_value=10, value=5, step=1)

        # Botão para iniciar o processamento
        if st.button('Processar Imagem'):
            # Executar SLIC
            sl = SLIC()
            mus, sets = sl.run(img_CIElab, k=k, iters=iters)
            clr_mask = sl.get_color_mask(mus, sets)

            # Mostrar a imagem processada
            st.image(clr_mask, caption='Imagem Processada', use_column_width=True)

if __name__ == "__main__":
    main()