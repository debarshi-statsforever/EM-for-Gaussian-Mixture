library(tiff)
library(pals) # RColorBrewer doesn't have 16-palettes
library(Rcpp)
sourceCpp('./calc_parallel.cpp')

### 3.b) i) read in the TIFF image and convert
### to a dataset of trivariate observations

load_data <- function() {
    x0 <- readTIFF(source="durga.tiff")
    shift <- 1
    newden <- 257
    newden <- max(255 + shift + 1, newden)
    x1 <- x0 + (shift/255)
    x2 <- x1 * (255/newden)
    x2
}

view_image <- function(img) {
    dev.new()
    grid::grid.raster(img)
}

image <- load_data()

#### 3.b) ii) use EM algorithm to perform model clustering and obtain IDs

em_algorithm <- function(data, pi_all, mu_all, sigma, NUM_STEPS=1, NUM_GROUPS=16) {
    h <- qnorm(data)
    # print(c(min(h), max(h)))
    w <- array(0, dim=c(dim(data)[1:2], NUM_GROUPS))
    return(em_algorithm_cpp(data, h, w, pi_all, mu_all, sigma, NUM_STEPS))
}

run_em <- function(img, NUM_STEPS=10, GROUPS=16) {
    rs <- cbind(sample(dim(img)[1], GROUPS), sample(dim(img)[2], GROUPS))
    mu_init <- matrix(0, nrow=GROUPS, ncol=3)
    for (i in 1:16) {
        mu_init[i,] <- qnorm(img[rs[i,1], rs[i,2], ])
    }
    pi_init <- runif(GROUPS, 0, 1)
    pi_init <- pi_init / sum(pi_init)
    sigma_init <- diag(runif(3, 0.01, 0.1))

    a0 <- proc.time()
    estim <- em_algorithm(data=img, pi_all=pi_init, mu_all=mu_init, sigma=sigma_init, NUM_STEPS=NUM_STEPS, NUM_GROUPS=GROUPS)
    b0 <- proc.time()
    print(b0-a0)
    save(estim, file=sprintf("EM_coeffs-%d.Rdata", NUM_STEPS))
    estim
}

load_estim <- function(steps) {
    fname <- sprintf("EM_coeffs-%d.Rdata", steps)
    load(fname,  temp_env <- new.env())
    env_list <- as.list(temp_env)
    env_list$estim
}

get_cluster_ids <- function(img, estim) {
    h <- qnorm(img)
    cluster_ids <- get_cids_cpp(img, h, estim$pi, estim$mu, estim$sigma)
    cluster_ids
}

### 3.b) iii) quantize image as per model, and view

get_quantized_image <- function(img, estim, cids) {
    mu <- estim$mu
    q_img <- pnorm(get_qimg_cpp(img, cids, mu))
    q_img
}

main <- function() {
    # estim <- run_em(image, 500)
    estim_loaded <- load_estim(500)
    cids <- get_cluster_ids(image, estim_loaded)
    q_image <- get_quantized_image(image, estim_loaded, cids)
    print(table(cids))
    image(t(cids[nrow(cids):1,]), col=glasbey(16))
    view_image(image)
    view_image(q_image)
    print(mean(abs(image-q_image)))
    estim_loaded
}
