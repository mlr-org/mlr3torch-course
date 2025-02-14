library(torch)
library(ggplot2)

run_adamw <- function(lr = 0.1, weight_decay = 0.01, epochs = 10, betas = c(0.9, 0.999), batch_size = 1, steps = epochs * (as.integer(ceiling(100 / batch_size)))) {
  w_true <- torch_tensor(0.4)
  b_true <- torch_tensor(0.7)
  data = with_torch_manual_seed({
    # DONT CHANGE 100
    X = torch_randn(100, 1) * 0.5
    Y = X * w_true + b_true + torch_randn(1000) * 2
    list(X = X, Y = Y)
  }, seed = 42)
  w <- torch_tensor(-1, requires_grad = TRUE)
  b <- torch_tensor(-1, requires_grad = TRUE)
  X = data$X
  Y = data$Y

  # Initialize optimizer
  opt <- optim_adamw(list(w, b), lr = lr, weight_decay = weight_decay, betas = betas)
  n <- X$size(1)

  # Initialize trajectory dataframe

  n_batches <- as.integer(ceiling(n / batch_size))

  trajectory <- data.frame(
    step = integer(epochs * n_batches + 1),
    w = numeric(epochs * n_batches + 1),
    b = numeric(epochs * n_batches + 1)
  )

  trajectory$step[1] = 0
  trajectory$w[1] = w$item()
  trajectory$b[1] = b$item()

  for (step in seq_len(steps)) {
    opt$zero_grad()
    batch <- step %% n_batches
    if (batch == 0) batch <- n_batches  # Adjust for zero modulus
    batch_start <- ((batch - 1) * batch_size + 1)
    batch_end <- min(batch * batch_size, n)
    x_batch <- X[batch_start:batch_end, , drop = FALSE]
    y_batch <- Y[batch_start:batch_end]

    # Forward pass
    y_pred <- x_batch * w + b
    loss <- torch_mean((y_pred - y_batch)^2)

    # Backward pass and optimization step
    loss$backward()
    opt$step()

    # Record trajectory
    trajectory$step[step + 1] <- step
    trajectory$w[step + 1] <- w$item()
    trajectory$b[step + 1] <- b$item()
  }

  a_range <- seq(-1.5, 1.5, length.out = 30)
  b_range <- seq(-1.5, 1.5, length.out = 30)

  # Create grid
  grid <- expand.grid(a = a_range, b = b_range)

  # Compute MSE for each grid point
  grid$mse <- apply(grid, 1, function(point) {
    a_val <- point["a"]
    b_val <- point["b"]
    y_pred <- X * a_val + b_val
    mse <- torch_mean((y_pred - Y)^2)$item()
    return(mse)
  })

  return(list(trajectory = trajectory, grid = grid))
}

# Start of Selection
.plot_contour = function(adamw_output) {
  grid <- adamw_output$grid
  p <- ggplot() +
    geom_contour(data = grid, aes(x = a, y = b, z = mse), bins = 15) +
    labs(x = "Weight (w)",
         y = "Bias (b)", level = NULL) +
    theme_minimal()
  return(p)
}
# End of Selection

.plot_segment = function(adamw_output) {
  trajectory <- adamw_output$trajectory
  grid <- adamw_output$grid
  geom_path(data = trajectory, aes(x = w, y = b), color = "red")
}


.plot_adamw_trajectory = function(adamw_output) {
  trajectory <- adamw_output$trajectory
  grid <- adamw_output$grid

  p <- .plot_contour(adamw_output)

  # Plot loss surface and trajectory
  p <- p +
    geom_point(data = trajectory[1, ], aes(x = w, y = b), color = "green", size = 3, shape = 17) +
    geom_point(data = trajectory[nrow(trajectory), ], aes(x = w, y = b), color = "purple", size = 3, shape = 17) +
    annotate("text", x = trajectory$w[1] + 0.1, y = trajectory$b[1], label = "Start", color = "green", hjust = 0) +
    annotate("text", x = trajectory$w[nrow(trajectory)] + 0.1, y = trajectory$b[nrow(trajectory)], label = "End", color = "purple", hjust = 0)

  # Additional segment added back
  p <- p +
    geom_path(data = trajectory, aes(x = w, y = b), color = "red")

  return(p)
}


plot_adamw_trajectories <- function(lr = 0.1, weight_decay = 0.01, epochs = 10, betas = c(0.9, 0.999), batch_size = 64, title = FALSE) {
  # create list with arguments
  args = list(lr = lr, weight_decay = weight_decay, epochs = epochs, betas = betas, batch_size = batch_size)
  is_list = sapply(args, is.list)
  if (sum(is_list) != 1) {
    stop("one of the arguments must be a list")
  }

  list_args = args[[which(is_list)]]
  list_arg_name = names(args)[is_list]
  other_args = args[!is_list]
  args = lapply(list_args, function(arg) {
    other_args[[list_arg_name]] = arg
    other_args
  })

  adamw_outputs <- lapply(args, function(arg) {
    do.call(run_adamw, arg)
  })

  # Generate the contour plot using the first output
  p <- .plot_contour(adamw_outputs[[1]])

  ps = list()

  # Add trajectories for each element in adamw_outputs
  i = 1
  for (output in adamw_outputs) {
    x <- args[[i]][[list_arg_name]]
    pi = p + .plot_segment(output)
    if (title) {
      pi = pi + ggplot2::ggtitle(paste0(list_arg_name, " = ", paste0(x, collapse = ", ")))
    }
    ps[[i]] = pi
    i = i + 1
  }

  cowplot::plot_grid(plotlist = ps, ncol = 2)
}

plot_2d_image <- function(image, min = NULL, max = NULL) {
  image <- as.matrix(image)
  tab <- as.table(image)
  rownames(tab) <- seq_len(nrow(image))
  colnames(tab) <- seq_len(ncol(image))
  tbl <- as.data.frame(tab)
  colnames(tbl) <- c("x", "y", "value")
  tbl$x <- as.integer(tbl$x)
  tbl$y <- as.integer(tbl$y)

  min = if (is.null(min)) NA else min
  max = if (is.null(max)) NA else max

  ggplot(tbl, aes(x = y, y = x, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black",
      limits = c(min, max)) +
    theme_void() +
    scale_y_reverse() +
    coord_fixed() +
    theme(legend.position = "none")
}
