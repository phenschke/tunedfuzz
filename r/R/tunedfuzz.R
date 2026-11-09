#' Edit costs
#'
#' A cost table maps edits of Unicode code points to costs. Each row is one
#' edit. `("R", "N", 0.22)` substitutes, `("n", "", 0.6)` deletes one character
#' and `("", "n", 0.6)` inserts it, and `("tz", "ß", 0.23)` replaces one string
#' with another, both at least one character long. In a symmetric table each row also gives the reverse edit. In a directed table a
#' row only replaces `a` in the first string of [tfz_distance()] by `b` in the
#' second. A later row for the same edit replaces an earlier one. Edits missing
#' from the table cost `default`. Costs are case-sensitive.
#'
#' @param rows A data frame with columns `a`, `b` and `cost`, or `NULL` for no
#'   edits (plain Levenshtein distance).
#' @param default Cost of every edit the table lacks.
#' @param symmetric Whether each row also gives the reverse edit.
#' @return A `tfz_costs` object. `as.data.frame()` gives back its rows. It
#'   survives [saveRDS()], and [tfz_write_costs()] saves it as CSV.
#' @examples
#' costs <- tfz_costs(data.frame(a = c("R", "tz"), b = c("N", "ß"), cost = c(0.22, 0.23)))
#' tfz_distance("Nußbaum", "Rutzbaum", costs)
#' @export
tfz_costs <- function(rows = NULL, default = 1, symmetric = TRUE) {
  if (is.null(rows)) rows <- data.frame(a = character(), b = character(), cost = numeric())
  new_costs(costs_new(utf8(rows$a), utf8(rows$b), as.numeric(rows$cost), default, symmetric))
}

#' @describeIn tfz_costs Writes the rows to a CSV file with columns `a`, `b`
#'   and `cost`, which `tfz_read_costs()` reads back. The file does not record
#'   `default` or `symmetric`.
#' @export
tfz_write_costs <- function(costs, path) {
  costs_write(costs_ptr(costs), path.expand(path))
  invisible(costs)
}

#' @describeIn tfz_costs Reads the `a`, `b` and `cost` columns of a CSV file.
#' @param path Path to a CSV file with a header.
#' @export
tfz_read_costs <- function(path, default = 1, symmetric = TRUE) {
  new_costs(costs_read(path.expand(path), default, symmetric))
}

#' @describeIn tfz_costs Loads a directed table shipped with the package,
#'   learned from public (truth, reading) pairs. The table edits truth into
#'   reading, so call `tfz_distance(truth, reading, costs)`, or build a
#'   `tfz_index()` of true names with `tfz_reverse_costs(costs)`. Use it as is,
#'   or as the `prior` of `tfz_learn_costs()`.
#' @param name Name of the pretrained table. `"general"` comes from every
#'   training corpus, `"antiqua"` and `"fraktur"` from the corpora printed in
#'   that type, and `"kurrent"` from HTR of German handwriting in Kurrent.
#' @export
tfz_pretrained_costs <- function(name = "general") {
  new_costs(costs_pretrained(name))
}

#' @describeIn tfz_costs Returns the table that edits the second string into
#'   the first, with every row turned around.
#' @param costs A `tfz_costs` object.
#' @export
tfz_reverse_costs <- function(costs) {
  new_costs(costs_reversed(costs_ptr(costs)))
}

#' Weighted edit distance
#'
#' Computes the cheapest cost of editing each `a` into the matching `b`. The
#' shorter argument is recycled.
#'
#' @param a,b Character vectors.
#' @param costs A [tfz_costs()] object.
#' @param score_cutoff Distances above it come back as `Inf`, and the
#'   computation stops early once every path is too expensive.
#' @param normalize Whether to divide each distance by the length of the
#'   longer string. Two empty strings are 0 apart. `score_cutoff` then applies
#'   to the normalized distance.
#' @param workers Threads to compute on; -1 uses all cores.
#' @return A numeric vector.
#' @export
tfz_distance <- function(a, b, costs, score_cutoff = Inf, normalize = FALSE, workers = 1) {
  n <- if (length(a) && length(b)) max(length(a), length(b)) else 0
  distance_pairs(rep_len(utf8(a), n), rep_len(utf8(b), n), costs_ptr(costs), score_cutoff, normalize,
                 as.integer(workers))
}

#' Edits behind a distance
#'
#' @inheritParams tfz_distance
#' @param a,b Two strings.
#' @return A data frame with columns `a`, `b` and `cost`, one row per edit in
#'   order, without matches. Its costs sum to `tfz_distance(a, b, costs)`.
#' @export
tfz_align <- function(a, b, costs) {
  as.data.frame(align_pair(utf8(a), utf8(b), costs_ptr(costs)))
}

#' Exact nearest-name search
#'
#' `tfz_index()` builds a trie of the names once. `tfz_extract_one()` and
#' `tfz_extract()` then search it for each query and return what brute force
#' over every name would return, in a fraction of the time. Ties go to the
#' lowest index. A name's cost is `tfz_distance(query, name, costs)`, so lower is
#' closer. The index
#' survives [saveRDS()] and rebuilds its trie on first use after [readRDS()].
#'
#' @param names Character vector of names to search.
#' @inheritParams tfz_distance
#' @return `tfz_index()`: a `tfz_index` object.
#' @export
tfz_index <- function(names, costs) {
  names <- utf8(names)
  ix <- structure(list(names = names, costs = costs, cache = new.env(parent = emptyenv())), class = "tfz_index")
  index_ptr(ix)
  ix
}

#' @describeIn tfz_index Returns the cheapest name for each query as a data
#'   frame with one row per query and columns `index`, `name` and `cost`, all
#'   `NA` where no name is within `score_cutoff`.
#' @param index A `tfz_index` object.
#' @param queries Character vector of queries.
#' @inheritParams tfz_distance
#' @export
tfz_extract_one <- function(index, queries, score_cutoff = Inf, normalize = FALSE, workers = 1) {
  hits <- tfz_extract(index, queries, limit = 1, score_cutoff = score_cutoff, normalize = normalize,
                      workers = workers)
  pos <- match(seq_along(queries), hits$query)
  data.frame(index = hits$index[pos], name = hits$name[pos], cost = hits$cost[pos])
}

#' @describeIn tfz_index Returns the `limit` cheapest names for each query, or
#'   all of them for `limit = Inf`, as a data frame with columns `query` (the
#'   query's position), `index`, `name` and `cost`, ordered by query, cost and
#'   index.
#' @param limit Names to return per query.
#' @export
tfz_extract <- function(index, queries, limit = 5, score_cutoff = Inf, normalize = FALSE, workers = 1) {
  limit <- if (is.infinite(limit)) -1L else as.integer(limit)
  hits <- index_extract(index_ptr(index), utf8(queries), limit, score_cutoff, normalize, as.integer(workers))
  data.frame(query = hits$query, index = hits$index, name = index$names[hits$index], cost = hits$cost)
}

#' Distance matrix
#'
#' @param queries,names Character vectors.
#' @inheritParams tfz_distance
#' @inheritParams tfz_extract
#' @return A matrix with one row per query and one column per name.
#' @export
tfz_cdist <- function(queries, names, costs, score_cutoff = Inf, normalize = FALSE, workers = 1) {
  d <- cdist_flat(utf8(queries), utf8(names), costs_ptr(costs), score_cutoff, normalize, as.integer(workers))
  matrix(d, nrow = length(queries), ncol = length(names), byrow = TRUE)
}

#' Learn edit costs from misread strings
#'
#' Learns edit costs from (truth, reading, count) pairs. Each round
#' aligns every misread pair under the current costs, counts the edits, and sets
#' `p = max(c - shrink * sqrt(c), 0) / n` and
#' `cost = min(1, log(p) / log(p_floor))`, rounded to 4 decimals, for an edit
#' seen `c` times, where `n` counts the occurrences of the edited strings in all
#' pairs. Candidate multi-character edits are the `seeds` and the 2:1, 1:2 and
#' 2:2 replacements difflib finds at least `min_multi` times.
#'
#' With `symmetric = FALSE`, each direction gets its own cost. An edit (`a`,
#' `b`) replaces `a` in the truth by `b` in the reading, and `n` counts the
#' occurrences of `a` in the truths (for an insertion, the places it could go).
#' Pass the result to `tfz_costs(table, symmetric = FALSE)`.
#'
#' A `prior` table, such as `tfz_pretrained_costs()`, shrinks each probability
#' towards the prior's `p0 = p_floor ^ cost`. The prior counts as `prior_weight`
#' occurrences, which gives
#' `p = (max(c - shrink * sqrt(c), 0) + prior_weight * p0) / (n + prior_weight)`.
#' So the prior's cost wins for an edit the data rarely shows, and the data's
#' count wins for an edit it shows often.
#'
#' @param truth,reading Character vectors of true strings and their readings.
#' @param count Times each pair occurs.
#' @param seeds Data frame (`a`, `b`) of multi-character edits to consider, such
#'   as doubled letters.
#' @param fixed Data frame (`a`, `b`, `cost`) of edits with a cost set by hand.
#'   The result includes them as given.
#' @param rounds Rounds of EM.
#' @param p_floor Edits this rare or rarer cost 1.
#' @param min_multi Times a short replacement must occur to become a candidate.
#' @param min_ratio Pairs with a difflib ratio below this are misalignments and
#'   left out.
#' @param shrink Standard errors taken off each count.
#' @param symmetric Whether to learn one cost per edit and its reverse.
#' @param prior A `tfz_costs` table to shrink towards. A directed prior needs
#'   `symmetric = FALSE`.
#' @param prior_weight Occurrences of each string the prior counts as.
#' @return A data frame with columns `a`, `b`, `count`, `p` and `cost`. It
#'   holds the edits that cost less than 1, by cost, then the `fixed` edits as
#'   given, with `count` and `p` `NA`. [tfz_costs()] takes it as it is.
#' @export
tfz_learn_costs <- function(truth, reading, count = 1, seeds = NULL, fixed = NULL, rounds = 4,
                           p_floor = 1e-5, min_multi = 20, min_ratio = 0.5, shrink = 2,
                           symmetric = TRUE, prior = NULL, prior_weight = 30) {
  n <- length(truth)
  prior_ptr <- if (is.null(prior)) NULL else costs_ptr(prior)
  if (is.null(seeds)) seeds <- data.frame(a = character(), b = character())
  if (is.null(fixed)) fixed <- data.frame(a = character(), b = character(), cost = numeric())
  t <- learn(utf8(truth), utf8(reading), rep_len(as.numeric(count), n),
             utf8(seeds$a), utf8(seeds$b), utf8(fixed$a), utf8(fixed$b), as.numeric(fixed$cost),
             as.integer(rounds), p_floor, min_multi, min_ratio, shrink, symmetric,
             prior_ptr, prior_weight)
  as.data.frame(t)
}

#' @export
print.tfz_costs <- function(x, ...) {
  kind <- if (x$symmetric) "symmetric" else "directed"
  cat("<tfz_costs:", nrow(x$rows), kind, "edits, default", x$default, ">\n")
  invisible(x)
}

#' @export
as.data.frame.tfz_costs <- function(x, ...) {
  x$rows
}

#' @export
print.tfz_index <- function(x, ...) {
  cat("<tfz_index:", length(x$names), "names>\n")
  invisible(x)
}

# saveRDS() drops an external pointer's address. Each R object keeps the data that rebuilds its
# pointer, so it survives the round trip.

new_costs <- function(ptr) {
  cache <- new.env(parent = emptyenv())
  cache$ptr <- ptr
  info <- costs_info(ptr)
  structure(list(rows = as.data.frame(info$rows), default = info$default,
                 symmetric = info$symmetric, cache = cache), class = "tfz_costs")
}

costs_ptr <- function(costs) {
  if (!inherits(costs, "tfz_costs")) stop("costs must be a tfz_costs object", call. = FALSE)
  if (is.null(costs$cache$ptr) || !ptr_alive(costs$cache$ptr)) {
    r <- costs$rows
    costs$cache$ptr <- costs_new(r$a, r$b, r$cost, costs$default, costs$symmetric)
  }
  costs$cache$ptr
}

index_ptr <- function(index) {
  if (is.null(index$cache$ptr) || !ptr_alive(index$cache$ptr)) {
    index$cache$ptr <- index_new(index$names, costs_ptr(index$costs))
  }
  index$cache$ptr
}

utf8 <- function(x) {
  x <- as.character(x)
  if (anyNA(x)) stop("strings must not be NA", call. = FALSE)
  enc2utf8(x)
}
