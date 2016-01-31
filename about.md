---
layout: page
title: About
permalink: /about/
---

{% for group in site.data.members %}
<div class ="row" style="margin-top=20px">
<div class="col-md-7" >
<h3>{{group.group}}</h3>
</div>
</div>
<div class ="row">
{% for member in group.people %}


<div class="col-md-3"><div><a href="{{url}}">
<img class="img-thumbnail" src="{% if member.githubid  %}https://avatars2.githubusercontent.com/u/{{member.githubid}}?v=3&s=400{% else %}{{member.pic}}{% endif %}"></a></div>
<h4>{{member.name}}</h4> </div>
<a href> </a>

{% endfor %}
</div>
{% endfor %}
