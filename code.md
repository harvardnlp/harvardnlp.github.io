---
layout: page
title: Code
permalink: /code/
---

The following are research projects that have developed into open-source libraries:

<div class ="row"></div>


<div style="text-align:center">
<h3 > Projects </h3>
</div>

<table class="table table-striped table-hover">
<tr><th> Title</th> <th>Link </th></tr>
{% for project in site.data.code %}
<tr><td>{{project.title}} </td><td><a href="{{project.link}}">GitHub</td></tr>
{% endfor %}
</table>




