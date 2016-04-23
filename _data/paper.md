---
layout: page
title: Publications
permalink: /papers/
---


{% for paper in site.data.papers %}

{{paper.name}}
{{paper.authors}}
{{paper.conference}}

{% endfor %}
