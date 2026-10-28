// Renders the species cards, lookalike pairs and training results from species.json / results.json.
(function () {
  "use strict";

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function tag(status) {
    return el("span", "tag " + status, status);
  }

  function card(s, byId) {
    var c = el("article", "card");
    c.dataset.status = s.status;
    var head = el("div", "head");
    var names = el("div");
    names.appendChild(el("h3", null, s.common));
    names.appendChild(el("div", "sci", s.scientific));
    head.appendChild(names);
    head.appendChild(tag(s.status));
    c.appendChild(head);
    c.appendChild(el("p", null, s.id_tips));
    if (s.lookalike && byId[s.lookalike]) {
      c.appendChild(el("p", "look", "Lookalike: " + byId[s.lookalike].common + " (" + byId[s.lookalike].status + ")"));
    }
    var meta = s.form + (s.fisc ? " · FISC Category " + s.fisc : "") + " · folder " + s.id;
    c.appendChild(el("div", "meta", meta));
    return c;
  }

  function side(s) {
    var d = el("div", "side " + s.status);
    var h = el("h3", null, s.common + " ");
    h.appendChild(tag(s.status));
    d.appendChild(h);
    d.appendChild(el("div", "sci", s.scientific));
    d.appendChild(el("p", null, s.id_tips));
    return d;
  }

  function renderSpecies(data) {
    var list = data.species;
    var byId = {};
    list.forEach(function (s) { byId[s.id] = s; });

    var grid = document.getElementById("species-grid");
    grid.textContent = "";
    // Invasives first: they're the ones people want to find.
    list.slice().sort(function (a, b) {
      return (a.status === b.status) ? a.common.localeCompare(b.common) : (a.status === "invasive" ? -1 : 1);
    }).forEach(function (s) { grid.appendChild(card(s, byId)); });

    var counts = { all: list.length, native: 0, invasive: 0 };
    list.forEach(function (s) { counts[s.status]++; });
    document.querySelectorAll("[data-count]").forEach(function (n) {
      n.textContent = "(" + counts[n.dataset.count] + ")";
    });

    var buttons = document.querySelectorAll(".filters button");
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        buttons.forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
        var f = b.dataset.filter;
        grid.querySelectorAll(".card").forEach(function (c) {
          c.hidden = f !== "all" && c.dataset.status !== f;
        });
      });
    });

    var pairs = document.getElementById("pairs");
    list.filter(function (s) { return s.status === "invasive" && s.lookalike && byId[s.lookalike]; })
      .forEach(function (inv) {
        var p = el("div", "pair");
        p.appendChild(side(byId[inv.lookalike]));
        p.appendChild(el("div", "vs", "vs."));
        p.appendChild(side(inv));
        pairs.appendChild(p);
      });
    return byId;
  }

  function pct(x) { return Math.round(x * 100) + "%"; }

  function renderResults(r, byId) {
    var body = document.getElementById("results-body");
    body.textContent = "";
    var stats = el("div", "stats");
    [[r.native_vs_invasive_accuracy, "native vs. invasive correct"],
     [r.species_accuracy, "exact species correct"]].forEach(function (s) {
      var d = el("div", "stat");
      d.appendChild(el("div", "num", pct(s[0])));
      d.appendChild(el("div", "lbl", s[1]));
      stats.appendChild(d);
    });
    var total = Object.keys(r.photos).reduce(function (a, k) { return a + r.photos[k]; }, 0);
    var d = el("div", "stat");
    d.appendChild(el("div", "num", String(total)));
    d.appendChild(el("div", "lbl", "photos, " + r.val_photos + " held out for testing"));
    stats.appendChild(d);
    body.appendChild(stats);

    var wrap = el("div", "table-wrap");
    var t = el("table");
    var thead = el("thead");
    var hr = el("tr");
    ["Species", "Status", "Photos", "Test accuracy", ""].forEach(function (h) { hr.appendChild(el("th", null, h)); });
    thead.appendChild(hr);
    t.appendChild(thead);
    var tb = el("tbody");
    Object.keys(r.per_class_accuracy).sort(function (a, b) {
      return r.per_class_accuracy[a] - r.per_class_accuracy[b];
    }).forEach(function (id) {
      var s = byId[id];
      var tr = el("tr");
      tr.appendChild(el("td", null, s ? s.common : id));
      var st = el("td");
      if (s) st.appendChild(tag(s.status)); else st.textContent = id;
      tr.appendChild(st);
      tr.appendChild(el("td", null, String(r.photos[id] || 0)));
      tr.appendChild(el("td", null, pct(r.per_class_accuracy[id])));
      var bt = el("td");
      var bar = el("div", "bar");
      var fill = el("span");
      fill.style.width = pct(r.per_class_accuracy[id]);
      bar.appendChild(fill);
      bt.appendChild(bar);
      tr.appendChild(bt);
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    body.appendChild(wrap);

    if (r.top_mixups && r.top_mixups.length) {
      body.appendChild(el("h3", null, "Most common mix-ups")).style.marginTop = "1.5rem";
      var ul = el("ul", "plain");
      r.top_mixups.slice(0, 5).forEach(function (m) {
        var a = byId[m.actual], p = byId[m.predicted];
        ul.appendChild(el("li", null, (a ? a.common : m.actual) + " taken for " +
          (p ? p.common : m.predicted) + " (" + m.count + "×)"));
      });
      body.appendChild(ul);
    }
    body.appendChild(el("p", "eyebrow", "Trained " + r.trained.replace("T", " ") + " · " + r.epochs + " epochs"));
  }

  fetch("species.json")
    .then(function (res) { return res.json(); })
    .then(function (data) {
      var byId = renderSpecies(data);
      return fetch("results.json")
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (r) { if (r) renderResults(r, byId); })
        .catch(function () { /* no results yet: keep the placeholder */ });
    })
    .catch(function () {
      document.getElementById("species-grid").innerHTML =
        '<p class="pending">Couldn\'t load species.json. Open the site through GitHub Pages or a local web server ' +
        '(<code>python -m http.server</code> in the docs folder), not by double-clicking the file.</p>';
    });
})();
