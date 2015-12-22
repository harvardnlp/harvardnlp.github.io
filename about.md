---
layout: page
title: About
permalink: /about/
---

{% for member in site.data.members %}

<div class ="row">
<div class="col-md-3 col-md-offset-3"><div><img class="img-thumbnail"></div>
<span class="lead">{{member.name}}</span><p>{{member.title}}</p> </div>
</div>
{% endfor %}
