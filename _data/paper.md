---
layout: page
title: Recent Papers
permalink: /papers/
---


{% for paper in site.data.papers %}

{{paper.name}}
{{paper.authors}}
{{paper.conference}}

{% endfor %}
