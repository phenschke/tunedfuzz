read_fixture <- function(f) {
  read.csv(test_path("fixtures", f), encoding = "UTF-8", colClasses = "character", na.strings = character())
}

toy <- tfz_read_costs(test_path("fixtures", "costs.csv"))

test_that("distances match the shared fixtures", {
  fx <- read_fixture("distances.csv")
  expect_equal(tfz_distance(fx$a, fx$b, toy), as.numeric(fx$expected))
})

test_that("costs round-trip through a data frame", {
  rows <- read_fixture("costs.csv")
  rows$cost <- as.numeric(rows$cost)
  expect_equal(as.data.frame(tfz_costs(rows)), rows)
  expect_error(tfz_costs(data.frame(a = "ab", b = "", cost = 0.5)), "deletion")
})

test_that("the score cutoff turns distances above it into Inf", {
  expect_equal(tfz_distance("Nußbaum", c("Rutzbaum", "Huber"), toy, score_cutoff = 0.5), c(0.45, Inf))
})

test_that("align explains the distance", {
  edits <- tfz_align("Nußbaum", "Rutzbaum", toy)
  expect_equal(edits, data.frame(a = c("N", "ß"), b = c("R", "tz"), cost = c(0.2, 0.25)))
})

test_that("the index equals brute force", {
  names <- c("Rodatus", "Robatus", "Moriß", "Müller", "Huber", "", "Robatus")
  queries <- c("Rodatus", "Moritz", "Mueller", "Xaver", "")
  ix <- tfz_index(names, toy)
  d <- tfz_cdist(queries, names, toy)
  for (q in seq_along(queries)) {
    want <- order(d[q, ], seq_along(names))
    got <- tfz_extract(ix, queries[q], limit = Inf, workers = 2)
    expect_equal(got$index, want)
    expect_equal(got$cost, d[q, want])
  }
  one <- tfz_extract_one(ix, queries, score_cutoff = 0.5)
  expect_equal(one$name, c("Rodatus", "Moriß", "Müller", NA, ""))
})

test_that("the learner recovers OCR confusions", {
  truth <- c("Rauch", "Rieger", "Seitz", "Moritz", "Lutz", "Huber", "Rotzer", "Kurz")
  read <- gsub("tz", "ß", sub("R", "N", truth))
  table <- tfz_learn_costs(c(truth, truth), c(truth, read), rep(c(50, 10), each = length(truth)),
                          fixed = data.frame(a = "ae", b = "ä", cost = 0.1), min_multi = 5)
  expect_named(table, c("a", "b", "count", "p", "cost"))
  cost <- setNames(table$cost, paste(table$a, table$b))
  expect_lt(cost[["N R"]], 0.5)
  expect_lt(cost[["tz ß"]], 0.5)
  expect_true(is.na(table$count[nrow(table)]))
  expect_equal(tfz_distance("Rotzer", "Noßer", tfz_costs(table)), cost[["N R"]] + cost[["tz ß"]])
})

test_that("directed tables cost each direction on its own", {
  costs <- tfz_costs(data.frame(a = c("R", "n", "tz"), b = c("N", "", "ß"), cost = c(0.2, 0.3, 0.25)),
                     symmetric = FALSE)
  expect_equal(tfz_distance(c("R", "Hans", "Seitz"), c("N", "Has", "Seiß"), costs), c(0.2, 0.3, 0.25))
  expect_equal(tfz_distance(c("N", "Has", "Seiß"), c("R", "Hans", "Seitz"), costs), c(1, 1, 2))
  back <- tfz_reverse_costs(costs)
  expect_equal(tfz_distance(c("N", "Has", "Seiß"), c("R", "Hans", "Seitz"), back), c(0.2, 0.3, 0.25))
  ix <- tfz_index(c("Seitz", "Seiß"), costs)
  expect_equal(tfz_extract(ix, "Seitz", limit = Inf)$cost, c(0, 0.25))
  expect_equal(tfz_extract(ix, "Seiß", limit = Inf)$cost, c(0, 2))
  expect_output(print(costs), "directed")
})

test_that("the learner learns each direction", {
  truth <- c("Rauch", "Rieger", "Seitz", "Moritz", "Lutz", "Huber", "Rotzer", "Kurz", "Nagl", "Bauß")
  read <- gsub("tz", "ß", sub("R", "N", truth))
  table <- tfz_learn_costs(c(truth, truth), c(truth, read), rep(c(50, 10), each = length(truth)),
                           min_multi = 5, symmetric = FALSE)
  cost <- setNames(table$cost, paste(table$a, table$b))
  expect_lt(cost[["R N"]], 0.5)
  expect_false("N R" %in% names(cost))
  costs <- tfz_costs(table, symmetric = FALSE)
  expect_equal(tfz_distance("Noßer", "Rotzer", costs), 3)
})

test_that("a pretrained table loads and serves as a prior", {
  general <- tfz_pretrained_costs()
  expect_false(general$symmetric)
  expect_gt(nrow(general$rows), 100)
  expect_lt(tfz_distance("Rauch", "Nauch", general), 1)
  expect_error(tfz_pretrained_costs("nonexistent"))
  expect_lt(tfz_distance("V", "B", tfz_pretrained_costs("fraktur")), 0.2)
  expect_gt(tfz_distance("V", "B", tfz_pretrained_costs("antiqua")), 0.2)
  expect_lt(tfz_distance("p", "g", tfz_pretrained_costs("kurrent")), 0.5)
  table <- tfz_learn_costs(c("Rauch", "Huber"), c("Nauch", "Huber"), c(3, 30),
                           symmetric = FALSE, prior = general, prior_weight = 1e9)
  cost <- setNames(table$cost, paste(table$a, table$b))
  prior <- setNames(general$rows$cost, paste(general$rows$a, general$rows$b))
  expect_equal(cost[["e c"]], prior[["e c"]], tolerance = 1e-3)
  expect_error(tfz_learn_costs("Rauch", "Nauch", prior = general))
})

test_that("tables round-trip through CSV", {
  path <- tempfile(fileext = ".csv")
  directed <- tfz_costs(data.frame(a = c("R", "a,b"), b = c("N", '"x"'), cost = c(0.2, 1 / 3)), symmetric = FALSE)
  for (costs in list(toy, directed)) {
    tfz_write_costs(costs, path)
    expect_equal(as.data.frame(tfz_read_costs(path, symmetric = costs$symmetric)), as.data.frame(costs))
  }
})

test_that("normalized distances divide by the longer string", {
  expect_equal(tfz_distance("Nußbaum", c("Rutzbaum", ""), toy, normalize = TRUE), c(0.45 / 8, 1))
  expect_equal(tfz_distance("Nußbaum", "Rutzbaum", toy, normalize = TRUE, score_cutoff = 0.05), Inf)
  names <- c("Rodatus", "Robatus", "Moriß", "Müller", "Huber", "")
  queries <- c("Rodatus", "Moritz", "Mueller", "")
  d <- tfz_cdist(queries, names, toy, normalize = TRUE)
  for (q in seq_along(queries)) expect_equal(d[q, ], tfz_distance(queries[q], names, toy, normalize = TRUE))
  got <- tfz_extract(tfz_index(names, toy), queries, limit = Inf, normalize = TRUE)
  expect_equal(got$cost, unlist(lapply(seq_along(queries), function(q) sort(d[q, ]))))
})

test_that("tables and indexes survive saveRDS", {
  path <- tempfile(fileext = ".rds")
  ix <- tfz_index(c("Nußbaum", "Huber"), toy)
  saveRDS(list(costs = toy, index = ix), path)
  back <- readRDS(path)
  expect_equal(tfz_distance("Nußbaum", "Rutzbaum", back$costs), 0.45)
  expect_equal(tfz_extract_one(back$index, "Rutzbaum")$name, "Nußbaum")
})
