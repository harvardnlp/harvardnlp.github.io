---
layout: page
title: RecentPapers
permalink: /papers/
---

{% for paper in site.data.papers %}

{{paper.name}}
{{paper.authors}}
{{paper.conference}}

{% endfor %}
