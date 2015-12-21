---
layout: page
title: About
permalink: /about/
---

{% for member in site.data.members %}

{{member.name}}
{{member.title}}

{% endfor %}
