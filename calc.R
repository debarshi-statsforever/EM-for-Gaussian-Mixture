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
    mu_init <- matrix(runif(48, -1, 1), nrow=GROUPS, ncol=3)
    pi_init <- rep(1/GROUPS, GROUPS)
    sigma_init <- diag(c(5, 5, 5))

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
    mu <- estim$mu
    h <- qnorm(img)
    cluster_ids <- get_cids_cpp(img, h, mu)
    cluster_ids
}

### 3.b) iii) quantize image as per model, and view

get_quantized_image <- function(img, estim, cids) {
    mu <- estim$mu
    q_img <- pnorm(get_qimg_cpp(img, cids, mu))
    q_img
}

main <- function() {
    estim <- run_em(image, 100)
    estim_loaded <- load_estim(100)
    cids <- get_cluster_ids(image, estim_loaded)
    q_image <- get_quantized_image(image, estim_loaded, cids)
    print(estim_loaded)
    image(t(cids[nrow(cids):1,]), col=glasbey(16))
    view_image(image)
    view_image(q_image)
}
