---
layout: page
title: Meetings
permalink: /meetings/
---


{% for term in site.data.meetings %}
{{term.term}}

| Date | Paper | Presenter | 
|------|-------|------------------|{% for paper in term.meetings %}
| {{ paper.date }}  | <a href="paper.link"> {{paper.paper }}</a> | {{paper.presenter}}          |{% endfor %}
{% endfor %}

