import numpy as np
import cv2
from scipy import ndimage
from skimage import color

class KMeans:
    """Kmeans para dados de imagem.
        Etapas de uso:
            1. Instancie
            2. Chame o método run e passe os argumentos conforme especificado
    """

    def init_centers(self, imgvol, k):
        """Inicialização de centro aleatório
        Args:
        imgvol: (np.array) volume da imagem, consistindo de CIElab e xy, Formato:[linhas, colunas, D=5]
        k: (int) número de clusters
        Return:
        mus: (np.array de inteiros) centros dos clusters, Formato: [K, 5]
        """

        rows, cols, D = imgvol.shape
        mus = np.zeros((k, D))

        mus_clr = np.random.randint(low=0, high=255, size=(k, 3))
        mus_rows = np.random.randint(low=0, high=rows, size=(k, 1))
        mus_cols = np.random.randint(low=0, high=cols, size=(k, 1))

        mus[:, :3] = mus_clr
        mus[:, 3:4] = mus_rows
        mus[:, 4:] = mus_cols

        return mus.astype(int)

    def prepare_img(self, img, include_xy=True):
        """Redimensiona a imagem, anexa coordenadas xy à imagem e converte para array numpy
        Args:
            img: (np.array ou cv) imagem de entrada de tipo desconhecido. Formato: [3, linhas, colunas]
        Returns:
            imgvol: (np.array float32) volume de imagem de saída com coordenadas yx anexadas. Formato: [linhas, colunas, 5]
        """

        r, c, _ = img.shape
        imgvol = np.zeros((r, c, 5))

        if (include_xy):
            x = np.arange(c)
            y = np.arange(r)
        else:
            x = np.arange(c) * 0
            y = np.arange(r) * 0


        cols, rows = np.meshgrid(x, y)

        imgvol[:,:,:3] = img
        imgvol[:,:,3] = rows
        imgvol[:,:,4] = cols

        return imgvol.astype(np.float32)

    def construct_sets(self, imgvol, mus):
        """Atribui pixels da imagem ao conjunto com centro mu mais próximo
        Args:
            imgvol: (np.array) volume da imagem, consistindo de CIElab e xy, Formato: [linhas, colunas, D=5]
        Returns:
            mask: (np.array) máscara inteira para atribuições de conjunto, Formato: [linhas, colunas, 1]
        """
        dists = self.compute_dists(imgvol, mus)

        mask = np.argmin(dists, axis = -1)

        return np.expand_dims(mask, -1)

    def compute_dists(self, imgvol, mus):
        """Calcula as distâncias entre pixels e centros.
        Args:
            imgvol: (np.array) imagem de entrada, Formato: [linhas, colunas, D=5]
            mus: (np.array) centros dos clusters, Formato: [K, 5]
        Returns:
            dists: (np.array) distâncias de saída, Formato: [linhas, colunas, K]
        """

        rows, cols, D = imgvol.shape
        K, _ = mus.shape
        dists2 = np.zeros((K, rows, cols))

        MUS = np.zeros((1,1,K, D))
        MUS[0,0,:,:] = mus
        IMG = np.zeros((rows, cols, 1, D))
        IMG[:,:,0,:] = np.copy(imgvol)

        dists2 = np.linalg.norm( IMG - MUS , axis = -1)

        self.dists = dists2

        return dists2

    def update_mus(selfs, imgvol, sets, K=2):
        """Atualiza os centros (mus) dada uma nova atribuição de conjunto.
        Args:
        imgvol: (np.array) volume da imagem, consistindo de CIElab e xy, Formato: [linhas, colunas, D=5]
        sets: (np.array) máscara inteira para atribuições de conjunto, Formato: [linhas, colunas, 1]
        K: (int) número de clusters
        Returns:
        new_mus: (np.array) centros dos clusters, Formato: [K, 5]
        """

        rows, cols, D = imgvol.shape
        new_mus = np.zeros((K, D))

        for k in range(K):
            logicMat = np.ones((rows,cols,1))*k == sets
            num_points = np.sum(logicMat)

            new_mus[k] = np.sum(np.sum(logicMat * imgvol, axis = 0), axis = 0)

            if (num_points > 0):
                new_mus[k] = new_mus[k]/ num_points

        return new_mus.astype(int)

    def get_color_mask(self, mus, sets, include_boundaries=False):
        """Obtém a máscara de cores, onde cada pixel recebe a cor do centro de cluster atribuído a ele.
        Args:
            mus: (np.array) centros dos clusters, Formato: [K, 5]
            sets: (np.array) máscara inteira para atribuições de conjunto, Formato: [linhas, colunas, 1]
        Returns:
            clr_mask: (np.array) máscara de cor, Formato: [linhas, colunas, 3]
        """
        rows, cols, _ = sets.shape
        K, _ = mus.shape

        clr_mask = np.zeros((rows, cols, 3))
        broad_sets = np.zeros((rows, cols, 3))
        broad_sets[..., :] = np.copy(sets)

        for i in range(len(mus)):
            lab = np.expand_dims(np.expand_dims(np.copy(mus[i, :3]), 0), 0)
            logicMat = broad_sets == i
            clr_mask =  clr_mask +  logicMat * lab

        rgb_clr_mask = color.lab2rgb(clr_mask)
        clr_mask_with_boundaries = rgb_clr_mask

        if(include_boundaries):

            gray_clr_mask = self.grayscale((rgb_clr_mask *255.0).astype(np.uint8))
            grad = self.gradxy(gray_clr_mask)
            grad[grad>0] = -1
            grad[grad == 0] = 1
            grad[grad == -1] = 0
            broad_grad = np.zeros((rows, cols, 3))
            broad_grad[..., :] = np.expand_dims(grad, -1)
            clr_mask_with_boundaries = rgb_clr_mask * broad_grad

        for k in range(K):
            y, x = mus[k, 3:]
            clr_mask_with_boundaries = cv2.circle(clr_mask_with_boundaries, center=(x, y), radius=1, color=(0,0,0) , thickness=2)

        return clr_mask_with_boundaries

    def res_error(self, mus1, mus2):
        """Calcula o erro residual médio, ou seja, E = ||mu1 - mu2|| / |mu1|
        Args:
            mus1: (np.array) centros de cluster anteriores, Formato: [K, D=5]
            mus2: (np.array) novos centros de cluster, Formato: [K, D=5]
        Returns:
            error: (float/int)
        """
        error = np.sum(np.linalg.norm(mus1 - mus2, axis=-1))

        return error/len(mus1)

    def run(self, img, k, iters=10):
        """Executa o K-Means.
        1. Inicializa os centros
        2. Constrói Conjuntos
        3. Atualiza mus (ou seja, recalcula os centros com base na pertinência aos conjuntos)
        4. Se convergiu, retorna mus e sets; caso contrário, volte para (2)
        Args:
        img: (np.array) imagem de entrada, Formato: [3, linhas, colunas]
        k: (int) número de clusters
        iters: (int) número de iterações
        Returns
        mus: (np.array) centros dos clusters, Formato: [K, 5]
        sets: (np.array) atribuições de clusters, Formato: [3, linhas, colunas]
        """

        imgvol = self.prepare_img(img)

        mus = self.init_centers(imgvol, k)

        for i in range(iters):

            sets = self.construct_sets(imgvol, mus)
            mus = self.update_mus(imgvol, sets, k)

            self.plot(imgvol[..., :3], mus, sets)

        return mus, sets

class SLIC(KMeans):

    def init_centers_old(self, imgvol, K):
        """Inicialização do Centro SLIC
        Args:
            imgvol: (np.array) volume da imagem, consistindo de CIElab e xy, Formato:[linhas, colunas, D=5]
            K: (int) número de clusters
        Return:
            mus: (np.array) centros dos clusters, Formato: [K, 5]
        """

        rows, cols, D = imgvol.shape
        N = rows*cols                       
        S = np.sqrt(rows*cols/K)          
        mus = np.zeros((K, D))


        k=0
        for i in range(0, int((rows/S)*cols) - int(S), int(S)):

            grid_idx = np.random.choice(np.arange(i+ int(S/3), i+ int(2*S/3) ))   
            grid_coords = self.idx2coord(grid_idx, cols)
            dist_to_center = min(S/2, (rows - grid_coords[0]*S) /2.0)
            center_coords = np.multiply(grid_coords, np.array([S, 1])) +  np.array([dist_to_center, 0])
            center_coords = center_coords.astype(int)

            if((center_coords > 3).all() and (center_coords < np.array([rows-3, cols- 3])).all() ):
                img_gray = self.grayscale(np.copy(imgvol[..., :3]))
                img_grad_mag = self.gradxy(img_gray)
                offset = np.array([center_coords[0]-3, center_coords[1]-3])
                window = img_grad_mag[center_coords[0]-3:center_coords[0]+4, center_coords[1]-3:center_coords[1]+4]
                window_idx = np.argmin(window)
                mus[k,3:]= offset + self.idx2coord(window_idx, 3)

            else:
                mus[k, 3:] = center_coords

            if (mus[k, 3]==375): import pdb; pdb.set_trace()
            k +=1

        for j in range(K):

            mus[j, :3] = imgvol[int(mus[j, 3]), int(mus[j, 4]), :3]


        return mus

    def init_centers(self, imgvol, K):
        """Inicialização do centro em intervalos igualmente espaçados S. (S = sqrt(N/S)). Não verifica gradientes como no artigo
        Args:
            imgvol: (np.array) volume da imagem, consistindo de CIElab e xy, Formato:[linhas, colunas, D=5]
            K: (int) número de clusters
        Returns:
            mus: (np.array) centros dos clusters, Formato: [K, 5]
        """
        img_h, img_w, D = imgvol.shape
        S = int(np.sqrt((img_h*img_w)/K))
        mus = np.zeros((K, D))


        h = S//2
        w = S//2
        i = 0

        while h < img_h:
            while (w < img_w and i<K):
                mus[i, :] = imgvol[h, w]
                i+=1
                w += S
            w = S // 2
            h += S
        return mus

    def compute_dists(self, imgvol, mus, m=20):
        """Calcula as distâncias entre pixels e centros, distância=infinito se a distância xy for > 2S
        Args:
        imgvol: (np.array) imagem de entrada, Formato: [linhas, colunas, D=5]
        mus: (np.array) centros dos clusters, Formato: [K, 5]
        m: (float) Hiperparâmetro, ponderação para proximidade espacial versus cor
        Returns:
        dists: (np.array) distâncias de saída, Formato:[linhas, colunas, K]
        """

        rows, cols, D = imgvol.shape
        K, _ = mus.shape
        S = np.sqrt(rows*cols/K)


        mus, imgvol = np.copy(mus).astype(np.float32), np.copy(imgvol).astype(np.float32)


        img_lab, img_yx= imgvol[..., :3], imgvol[..., 3:] 


        dists_yx = np.ones((rows, cols, K, 2)) * np.inf
        dists_lab = np.ones((rows, cols, K, 3)) * np.inf

        mu_lab_broadcast = np.zeros((1, 1, 1, 3))
        mu_yx_broadcast = np.zeros((1, 1, 1,2))

        for j in range(K):
            mu_lab_broadcast[..., :] = mus[j, :3]
            mu_yx_broadcast[..., :] = mus[j, 3:]

            center_r, center_c = mus[j, 3:]

            r1 = max(0, center_r  -  S)
            r2 = min(center_r + S+1, rows)
            c1 = max(0, center_c - S)
            c2 = min(center_c + S+1, cols)

            r1, r2, c1, c2 = int(r1), int(r2), int(c1), int(c2)

            dists_lab[r1:r2, c1:c2, j, :] = np.abs(img_lab[r1:r2, c1:c2] - mu_lab_broadcast ) 
            dists_yx[r1:r2, c1:c2, j, :] = np.abs(img_yx[r1:r2, c1:c2] - mu_yx_broadcast) 

        dists_lab = np.linalg.norm(dists_lab, axis=-1) 
        dists_yx = np.linalg.norm(dists_yx,  axis=-1) 


        dists = np.zeros((rows, cols, K, 2))

        dists[..., 0] = dists_lab/m
        dists[..., 1] = dists_yx/S
        dists = np.linalg.norm(dists,  axis=-1)

        self.dists = dists

        return dists

    def compute_dists_vectorised(self, imgvol, mus, m=20):
        """Calcula as distâncias entre pixels e centros, distância=infinito se a distância xy for > 2S
        Args:
        imgvol: (np.array) imagem de entrada, Formato: [linhas, colunas, D=5]
        mus: (np.array) centros dos clusters, Formato: [K, 5]
        m: (float) Hiperparâmetro, ponderação para proximidade espacial versus cor
        Returns:
        dists: (np.array) distâncias de saída, Formato:[linhas, colunas, K]
        """

        rows, cols, D = imgvol.shape
        K, _ = mus.shape
        S = np.sqrt(rows*cols/K) # Stride


        MUS_lab = np.zeros((1,1,K, 3))
        MUS_yx = np.zeros((1,1,K, 2))
        IMG_lab, IMG_yx = np.zeros((rows, cols, 1, 3)), np.zeros((rows, cols, 1, 2))

        IMG_lab[:,:, 0, :], IMG_yx[:,:, 0, :]= np.copy(imgvol[..., :3]), np.copy(imgvol[..., 3:])
        MUS_lab[0,0,:,:] = mus[:,:3]
        MUS_yx[0,0,:,:] = mus[:,3:]

        dists_lab = np.linalg.norm( IMG_lab - MUS_lab , axis = -1)  
        dists_yx = np.abs(IMG_yx - MUS_yx)                         
        dists_yx[dists_yx > S] = np.inf
        dists_yx = np.linalg.norm( dists_yx , axis = -1)   

        dists = np.zeros((rows, cols, K, 2))
        dists[..., 0] = dists_lab/m
        dists[..., 1] = dists_yx/S
        dists = np.linalg.norm(dists,  axis=-1)


        self.dists = dists

        return dists

    def idx2coord(self, index, cols):
        """Converte um índice em uma coordenada
        Args:
        index: (int)
        cols: int
        Returns:
        coords: (np.array) array 1D contendo coordenadas
        """
        col = index % cols
        row = int(index/cols)
        coords = np.array([row, col])

        return coords

    def gradxy(self, img, sigma=2, smooth=False):
        """Cálculo do Gradiente da Imagem por convolução. Imagem suavizada antes do cálculo do gradiente
        Args:
        img: (np array) Imagem de entrada em escala de cinza, Formato [Linhas, Colunas]
        sigma: (int) desvio padrão para o filtro gaussiano.
        Returns:
        img_grad_mag: (np array) magnitude do gradiente da imagem, Formato: [Linhas, Colunas]
        """

        rows, cols = img.shape
        img_grad = np.zeros((rows, cols, 2))

        fx = np.array([[1,-1]])
        fy = np.array([[1],[-1]])

        img_new = np.copy(img)
        if(smooth):
            img_new = self.convolve(np.copy(img), self.gaus(sigma))

        Ix = self.convolve(img_new, fx)
        Iy = self.convolve(img_new, fy)


        img_grad[..., 0], img_grad[..., 1] = Ix, Iy
        img_grad_mag = np.linalg.norm(img_grad, axis=-1)

        return img_grad_mag

    def convolve(self, img, filter):
        """Operação de convolução
        Args:
        img: (np.array) imagem de entrada, Formato [linhas, colunas]
        Returns:
        imagem convoluída do mesmo tamanho
        """

        return ndimage.convolve(img,  filter, mode='constant')

    def gaus(self, sigma):
        """Cria um kernel gaussiano para convolução
        Args:
            sigma: (float) desvio padrão da gaussiana
        Returns:
            gFilter: (np.array) filtro, Formato: [5,5]
        """
        kSize = 5
        gFilter = np.zeros((kSize, kSize))
        gausFunc = lambda u,v, sigma : (1/(2*np.pi*(sigma**2))) * np.exp( - ( (u**2) + (v**2) )/ (2 * (sigma**2)) )

        centerPoint = kSize//2

        for i in range(kSize):
            for j in range(kSize):

                u = i - centerPoint
                v = j - centerPoint

                gFilter[i, j] = gausFunc(u,v,sigma)


        return gFilter

    def grayscale(self, img):
        """Conversão de RGB para escala de cinza
        Args:
            img: (np array) imagem de entrada, Formato [3, Linhas, Colunas]
        Returns:
            img_gray: (np array) imagem de entrada, Formato [Linhas, Colunas]
        """

        img_gray = img

        # Omit if already grayscale
        if(len(img.shape) > 2):
            img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img_gray


    def run(self, img, k, iters=10):
        """Executa o SLIC. Resumo do algoritmo:
        1. Inicializa centros em um padrão de grade em intervalos de S
        2. Constrói conjuntos usando medida de distância
        3. Atualiza mus (ou seja, recalcula centros com base nas pertinências aos conjuntos)
        4. Se convergiu, retorna mus e sets; caso contrário, volta para (2)
        Args:
        img: (np.array) imagem de entrada, Formato: [3, linhas, colunas]
        k: (int) número de clusters
        iters: (int) número de iterações
        Returns
        mus: (np.array) centros dos clusters, Formato: [K, 5]
        sets: (np.array) atribuições de clusters, Formato: [3, linhas, colunas]
        """
        imgvol = self.prepare_img(img)

        mus = self.init_centers(imgvol, k)

        for i in range(iters):

            sets = self.construct_sets(imgvol, mus)
            old_mus = np.copy(mus)
            mus = self.update_mus(imgvol, sets, k)
            error = self.res_error(np.copy(old_mus), np.copy(mus))
            print("iter: {} Error: {}".format(i, error))

            if(error < 3):
                break

        return mus, sets