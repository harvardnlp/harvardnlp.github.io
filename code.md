---
layout: page
title: Code
permalink: /code/
---

{% for project in site.data.code %}
[{{project.title}}]({{project.link}})
{% endfor %}
